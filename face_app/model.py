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
"""Age-cGAN Model.

Dataset Citation:
@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')

from scipy.io import loadmat
from datetime import datetime
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Upsampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, add, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import Lambda, Reshape, Flatten, concatenate, Dropout

DATASET_PATH = "./wiki_crop/"

def compute_age(photo_date, dob):
    """Calculates the age from the dob and the date of photo taken.
    """
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    if birth.month < 7:
        return photo_date - birth.year
    else:
        return photo_date - birth.year - 1

def load_data(path):
    """Loads the image paths and the calculated ages for each photo. 
    """
    metadata = loadmat(os.path.join(path, "wiki.mat"))
    paths = metadata['wiki'][0, 0]['full_path'][0]
    dob = meta['wiki'][0, 0]['dob'][0]
    photo_date = metadata['wiki'][0, 0]['photo_taken'][0]
    calculated_age = [compute_age(photo_date[i], dob[i]) for i in range(len(dob))]

    images = []
    ages_list = []

    for i, image_path in enumerate(paths):
        images.append(image_path[0])
        ages_list.append(calculated_age[i])

    return images, ages_list

def encoder():
    """Builds the Encoder network.
    """
    input_layer = Input(shape=(64, 64, 3))
    x = Conv2D(32, (5,5), strides=2, padding='same')(input_layer)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (5,5), strides=2, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (5,5), strides=2, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (5,5), strides=2, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(100)(x)

    return Model(inputs=[input_layer], outputs=[x])


def generator():
    """Builds the Generator network.
    """
    latent_vector = Input(shape=(100,))
    conditioning_variable = Input(shape=(6,))

    x = concatenate([latent_vector, conditioning_variable])

    x = Dense(20148, input_dim=106)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(16384)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)

    x = Upsampling2D(size=(2,2))(x)
    x = Conv2D(128, (5,5), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = Upsampling2D(size=(2,2))(x)
    x = Conv2D(64, (5,5), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = Upsampling2D(size=(2,2))(x)
    x = Conv2D(64, (5,5), padding='same')(x)
    x = Activation('tanh')(x)

    return Model(inputs=[latent_vector, conditioning_variable], outputs=[x])

 def expand_dims(label):
    label = tf.keras.backend.expand_dims(label, 1)
    label = tf.keras.backend.expand_dims(label, 1)
    return tf.keras.backend.tile(label, [1, 32, 32, 1])

def discriminator():
    """Builds the Discriminator network.
    """
    image = Input(shape=(64, 64, 3))
    label = Input(shape=(6,))

    x = Conv2D(64, (3,3), strides=2, padding='same')(image)
    x = LeakyReLU(0.2)(x)

    label = Lambda(expand_dims)(label)
    
    x = concatenate([x, label], axis=3)
    x = Conv2D(128, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[images, label], outputs=[x])


def run_main(argv):
    del argv
    kwargs = {'path' : DATASET_PATH}
    main(**kwargs)  

def main(path):
    images, age_list = load_data(path)

if __name__ == '__main__':
    app.run(run_main)