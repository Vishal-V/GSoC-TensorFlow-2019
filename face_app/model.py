from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Upsampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, add, ZeroPadding2D, LeakyReLU

DATASET_PATH = ''
IMG_WIDTH = 128
IMG_HEIGHT = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, shape):
        self.scale = self.add_weight(name='scale', shape=shape[-1:], initializer=tf.keras.initializers.RandomNormal(0.0, 0.002), trainable=True)
        self.offset = self.add_weight(name='offset', shape=shape[-1:], initializer='zeros', trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axis=1, keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        x = (inputs - mean) * inv
        return self.scale * x + self.offset

def load_data(path):
    pass



class CycleGAN(object):
    def __init__():
        pass

    def train():
        pass
        
def run_main(argv):
    del argv
    kwargs = {'path' : DATASET_PATH}
    main(**kwargs)

def main(path):
    pass

if __name__ == '__main__':
    app.run(run_main)