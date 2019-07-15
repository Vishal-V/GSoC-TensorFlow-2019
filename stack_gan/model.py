# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""StackGAN.
"""

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

assert tf.__version__.startswith('2')

from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyRelu, BatchNormalization, Relu, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense
from tensorflow.keras.layers import Flatten, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


############################################################
# Conditioning Augmentation Network
############################################################

def conditioning_augmentation(x):
	"""The mean_logsigma passed as argument is converted into the text conditioning variable.

	Args:
		x: The output of the text embedding passed through a FC layer with LeakyRelu non-linearity.

	Returns:
	 	c: The text conditioning variable after computation.
	"""
	mean = x[:, :128]
	log_sigma = x[:, 128:]

	stddev = tf.keras.math.exp(log_sigma)
	epsilon = K.random_normal(shape=K.constant((mean.shape[1], ), dtype='int32'))
	c = mean + stddev * epsilon
	return c

def build_ca_network():
	"""Builds the conditioning augmentation network.
	"""
	input_layer1 = Input(shape=(1024,))
	mls = Dense(256)(input_layer1)
	mls = LeakyRelu(alpha=0.2)(mls)
	ca = lambda(conditioning_augmentation)(mls)
	return Model(inputs=[input_layer1], outputs=[ca])


############################################################
# Stage 1 Generator Network (CGAN)
############################################################

def UpSamplingBlock(x, num_kernels):
	"""An Upsample block with Upsampling2D, Conv2D, BatchNormalization and a Relu activation.

	Args:
		x: The preceding layer as input.
		num_kernels: Number of kernels for the Conv2D layer.

	Returns:
		x: The final activation layer after the Upsampling block.
	"""
	x = UpSampling2D(size=(2,2))(x)
	x = Conv2D(num_kernels, kernel_size=(3,3), padding='same', strides=1, use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = Relu()(x)
	return x

def build_stage1_generator():
	"""Build the Stage 1 Generator Network using the conditioning text and latent space

	Returns:
		Stage 1 Generator Model for StackGAN.
	"""
	input_layer1 = Input(shape=(1024,))
	ca = Dense(256)(input_layer1)
	ca = LeakyRelu(alpha=0.2)(ca)

	# Obtain the conditioned text
	c = lambda(conditioning_augmentation)(ca)

	input_layer2 = Input(shape=(100,))
	concat = Concatenate(axis=1)([c, input_layer2])

	x = Dense(16384, use_bias=False)(conact)
	x = Relu()(x)
	x = Reshape((4, 4, 1024), input_shape=(16384,))(x)

	x = UpSamplingBlock(x, 512)
	x = UpSamplingBlock(x, 256)
	x = UpSamplingBlock(x, 128)
	x = UpSamplingBlock(x, 64)

	x = Conv2D(3, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
	x = Activation('tanh')(x)

	stage1_gen = Model(inputs=[input_layer1, input_layer2], outputs=[x, ca])
	return stage1_gen


############################################################
# Stage 1 Discriminator Network
############################################################	

def ConvBlock(x, num_kernels, kernel_size=(4,4), strides=2, activation=True):
	"""A ConvBlock with a Conv2D, BatchNormalization and LeakyRelu activation.

	Args:
		x: The preceding layer as input.
		num_kernels: Number of kernels for the Conv2D layer.

	Returns:
		x: The final activation layer after the ConvBlock block.
	"""
	x = Conv2D(num_kernels, kernel_size=kernel_size, padding='same', strides=strides, use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	
	if activation:
		x = LeakyRelu(alpha=0.2)(x)
	return x

def build_embedding_compressor():
    """Build embedding compressor model
    """
    input_layer1 = Input(shape=(1024,))
    x = Dense(128)(input_layer1)
    x = ReLU()(x)

    model = Model(inputs=[input_layer1], outputs=[x])
    return model

def build_stage1_discriminator():
	"""Builds the Stage 1 Discriminator that uses the 64x64 resolution images from the generator
	and the compressed and spatially replicated embedding.

	Returns:
		Stage 1 Discriminator Model for StackGAN.
	"""
	input_layer1 = Input(shape=(64, 64, 3))

	x = Conv2D(64, kernel_size=(4,4), strides=2, padding='same', use_bias=False)(input_layer1)
	x = LeakyRelu(alpha=0.2)(x)

	x = ConvBlock(x, 128)
	x = ConvBlock(x, 256)
	x = ConvBlock(x, 512)

	# Obtain the compressed and spatially replicated text embedding
	input_layer2 = Input(shape=(4, 4, 128))
	concat = concatenate([x, input_layer2])

	x1 = Conv2D(512, kernel_size=(1,1), padding='same', strides=1, use_bias=False)(concat)
	x1 = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x1 = LeakyRelu(alpha=0.2)(x)

	# Flatten and add a FC layer to predict.
	x1 = Flatten()(x1)
	x1 = Dense(1)(x1)
	x1 = Activation('sigmoid')(x1)

	stage1_dis = Model(inputs=[input_layer1, input_layer2], outputs=[x1])
	return stage1_dis
	

############################################################
# Adversarial Model
############################################################

def build_adversarial(generator_model, discriminator_model):
	"""Adversarial model.

	Args:
		generator_model: Stage 1 Generator Model
		discriminator_model: Stage 1 Discriminator Model

	Returns:
		Adversarial Model.
	"""
	input_layer1 = Input(shape=(1024,))
	input_layer2 = Input(shape=(100,))
	input_layer3 = Input(shape=(4, 4, 128))

	x, ca = generator_model([input_layer1, input_layer2])

	discriminator_model.trainable = False

	probabilities = discriminator_model([x, input_layer3])

	adversarial_model = Model(inputs=[input_layer1, input_layer2, input_layer3], outputs=[probabilities, ca])
	return adversarial_model


############################################################
# Stage 2 Generator Network
############################################################

def concat_along_dims(inputs):
	"""Joins the conditioned text with the encoded image along the dimensions.

	Args:
		inputs: consisting of conditioned text and encoded images as [c,x].

	Returns:
		Joint block along the dimensions.
	"""
	c = inputs[0]
	x = inputs[1]

	c = K.expand_dims(c, axis=1)
	c = K.expand_dims(c, axis=1)
	c = K.tile(c, [1, 16, 16, 1])
	return K.concatenate([c, x], axis = 3)

def residual_block(inputs):
	"""Residual block with plain identity connections.

	Args:
		inputs: input layer or an encoded layer

	Returns:
		Layer with computed identity mapping.
	"""
	x = Conv2D(512, kernel_size=(3,3), padding='same', use_bias=False)(inputs)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = Relu()(x)
	
	x = Conv2D(512, kernel_size=(3,3), padding='same', use_bias=False)(inputs)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	
	x = add([x, inputs])
	x = Relu()(x)

	return x

def build_stage2_generator():
	"""Build the Stage 2 Generator Network using the conditioning text and images from stage 1.

	Returns:
		Stage 2 Generator Model for StackGAN.
	"""
	input_layer1 = Input(shape=(1024,))
	input_images = Input(shape=(64, 64, 3))

	# Conditioning Augmentation
	ca = Dense(256)(input_layer1)
	ca = LeakyRelu(alpha=0.2)(ca)
	c = lambda(conditioning_augmentation)(ca)

	# Downsampling block
	x = ZeroPadding2D(padding=(1,1))(input_images)
	x = Conv2D(128, kernel_size=(3,3), strides=1, use_bias=False)(x)
	x = Relu()(x)

	x = ZeroPadding2D(padding=(1,1))(x)
	x = Conv2D(256, kernel_size=(4,4), strides=2, use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = Relu()(x)

	x = ZeroPadding2D(padding=(1,1))(x)
	x = Conv2D(512, kernel_size=(4,4), strides=2, use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = Relu()(x)

	# Concatenate text conditioning block with the encoded image
	concat = concat_along_dims([c, x])

	# Residual Blocks
	x = ZeroPadding2D(padding=(1,1))(concat)
	x = Conv2D(512, kernel_size=(3,3), padding='same', use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = Relu()(x)

	x = residual_block(x)
	x = residual_block(x)
	x = residual_block(x)
	x = residual_block(x)

	# Upsampling Blocks
	x = UpSamplingBlock(x, 512)
	x = UpSamplingBlock(x, 256)
	x = UpSamplingBlock(x, 128)
	x = UpSamplingBlock(x, 64)

	x = Conv2D(3, kernel_size=(3,3), padding='same', use_bias=False)(x)
	x = Activation('tanh')(x)

	stage2_gen = Model(inputs=[input_layer1, input_images], outputs=[x, ca])
	return stage2_gen


############################################################
# Stage 2 Discriminator Network
############################################################

def build_stage2_discriminator():
	"""Builds the Stage 2 Discriminator that uses the 256x256 resolution images from the generator
	and the compressed and spatially replicated embeddings.

	Returns:
		Stage 2 Discriminator Model for StackGAN.
	"""
	input_layer1 = Input(shape=(256, 256, 3))

	x = Conv2D(64, kernel_size=(4,4), padding='same', strides=2, use_bias=False)(input_layer1)
	x = LeakyRelu(alpha=0.2)(x)

	x = ConvBlock(x, 128)
	x = ConvBlock(x, 256)
	x = ConvBlock(x, 512)
	x = ConvBlock(x, 1024)
	x = ConvBlock(x, 2048)
	x = ConvBlock(x, 1024, (1,1), 1)
	x = ConvBlock(x, 512, (1,1), 1, False)

	x1 = ConvBlock(x, 128, (1,1), 1)
	x1 = ConvBlock(x1, 128, (3,3), 1)
	x1 = ConvBlock(x1, 128, (3,3), 1, False)

	x2 = add([x, x1])
	x2 = LeakyRelu(alpha=0.2)(x2)

	# Concatenate compressed and spatially replicated embedding
	input_layer2 = Input(shape=(4, 4, 128))
	concat = concatenate([x2, input_layer2])

	x3 = Conv2D(512, kernel_size=(1,1), strides=1, padding='same')(concat)
	x3 = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x3)
	x3 = LeakyRelu(alpha=0.2)(x3)

	# Flatten and add a FC layer
	x3 = Flatten()(x3)
	x3 = Dense(1)(x3)
	x3 = Activation('sigmoid')(x3)

	stage2_dis = Model(inputs=[input_layer1, input_layer2], outputs=[x3])
	return stage2_dis

############################################################
# Train Utilities
############################################################

def checkpoint_prefix():
	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

	return checkpoint_prefix

def adversarial_loss(y_true, y_pred):
	mean = y_pred[:, :128]
	ls = y_pred[:, 128:]
	loss = -ls + 0.5 * (-1 + tf.math.exp(2.0 * ls) + tf.math.square(mean))
	loss = tf.math.mean(loss)
	return loss

def load_data():
	# TODO: load the data and labels from the CUB birds folder
	x = []
	y = []
	embeds = []

	return x, y, embeds


############################################################
# StackGAN class
############################################################

# TODO: Stage 1 Gen LR Decay
class StackGanStage1(object):
	"""StackGAN Stage 1 class.

	Args:
		epochs: Number of epochs
		z_dim: Latent space dimensions
		batch_size: Batch Size
		enable_function: If True, training function is decorated with tf.function
		stage1_generator_lr: Learning rate for stage 1 generator
		stage1_discriminator_lr: Learning rate for stage 2 generator
	"""
	def __init__(self, epochs=1000, z_dim=100, enable_function=True, stage1_generator_lr=0.0002, stage1_discriminator_lr=0.0002):
		self.epochs = epochs
		self.z_dim = z_dim
		self.enable_function = enable_function
		self.stage1_generator_lr = stage1_generator_lr
		self.stage1_discriminator_lr = stage1_discriminator_lr
		self.image_size = 64
		self.conditioning_dim = 128
		self.batch_size = batch_size
		self.stage1_generator_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)
		self.stage1_discriminator_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
		self.stage1_generator = build_stage1_generator()
		self.stage1_generator.compile(loss='mse', optimizer=stage1_generator_optimizer)
		self.stage1_discriminator = build_stage1_discriminator()
		self.stage1_discriminator.compile(loss='binary_crossentropy', optimizer=stage1_discriminator_optimizer)
		self.ca_network = build_ca_network()
		self.ca_network.compile(loss='binary_crossentropy', optimizer='Adam')
		self.embedding_compressor = build_embedding_compressor()
		self.embedding_compressor.compile(loss='binary_crossentropy', optimizer='Adam')
		self.stage1_adversarial = build_adversarial()
		self.stage1_adversarial.compile(loss=['binary_crossentropy', 'adversarial_loss'], loss_weights=[1, 2.0], optimizer=stage1_generator_optimizer)
		self.checkpoint1 = tf.train.Checkpoint(
        	generator_optimizer=self.stage1_generator_optimizer,
        	discriminator_optimizer=self.stage1_discriminator_optimizer,
        	generator=self.stage1_generator,
        	discriminator=self.stage1_discriminator)

	def visualize():
		"""Only for testing
		"""
		tb = TensorBoard(log_dir="logs/".format(time.time()))
	    tb.set_model(self.stage1_generator)
	    tb.set_model(self.stage1_discriminator)
	    tb.set_model(self.ca_network)
	    tb.set_model(self.embedding_compressor)

	def train_stage1():
		"""Trains the stage1 StackGAN.
		"""

		x_train, y_train, train_embeds = load_data()
		x_test, y_test, test_embeds = load_data()

		real = np.ones((self.batch_size, 1), dtype='float') * 0.9
		fake = np.zeros((self.batch_size, 1), dtype='float') * 0.1

		for epoch in range(self.epochs):
			print(f'Epoch: {epoch}')
 
         	gen_loss = []
        	dis_loss = []

        	num_batches = x_train[0] / self.batch_size

        	for i in range(num_batches):
        		print(f'Batch: {i+1}')

        		latent_space = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        		embedding_text = train_embeds[i * self.batch_size:(i + 1) * self.batch_size]
        		compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)

        		g_loss = self.stage1_adversarial.train_on_batch([embedding_text, latent_space, compressed_embedding],
        			[K.ones((batch_size, 1)) * 0.9, K.ones((batch_size, 256)) * 0.9])

            	print(f'Generator Loss:{g_loss}')
            	gen_loss.append(g_loss)
