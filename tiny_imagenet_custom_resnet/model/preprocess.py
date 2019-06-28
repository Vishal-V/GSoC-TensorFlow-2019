import os
import tensorflow
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import img_to_array

class EpochCheckpoint(Callback):
	def __init__(self, every=5, startAt=0):
		super(Callback, self).__init__()
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# Save the model to disk at 'every' interval
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath,
				"epoch_{}.hdf5".format(self.intEpoch + 1)])
			self.model.save(p, overwrite=True)

		self.intEpoch += 1


class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# Rearranges the dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)
