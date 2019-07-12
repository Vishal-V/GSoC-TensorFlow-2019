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

def build_ca_model():
	"""Builds a conditioning augmentation network.

	Returns: 
		Model with input_layer as input and text conditioning variable as output.
	"""
	input_layer = Input(shape=(1024,))
	x = Dense(256)(input_layer)
	x = LeakyRelu(alpha=0.2)(x)

	c = lambda(conditioning_augmentation)(x)
	return Model(inputs=[input_layer], outputs=[c])


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
	x = Conv2D(num_kernels, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = Relu()(x)
	return x

def build_stage1_generator():
	"""Build the Stage 1 Generator Network using the conditioning text and latent space

	Returns:
		Stage 1 Generator Model
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

def ConvBlock(x, num_kernels):
	x = Conv2D(num_kernels, kernel_size=(4,4), padding='same', strides=2, use_bias=False)(x)
	x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
	x = LeakyRelu(alpha=0.2)(x)
	return x

def build_stage1_discriminator():
	"""Builds the Stage 1 Discriminator that uses the 64x64 resolution images from the generator
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

	# Flatten and add a FC layer
	x1 = Flatten()(x1)
	x1 = Dense(1)(x1)
	x1 = Activation('sigmoid')(x1)

	stage1_dis = Model(inputs=[input_layer1, input_layer2], outputs=[x1])
	return stage1_dis

