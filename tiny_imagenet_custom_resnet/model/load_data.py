import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

VAL_ANNOT = "/tiny-imagenet-200/val/val_annotations.txt"
TRAIN = "/content/tiny-imagenet-200/train/"
VAL = "/content/tiny-imagenet-200/val/images"

class LoadTinyImageNet(img_size=64, train_size=10000, val_size=1000):
	def __init__(self, img_size, train_size, val_size):
		self.img_size = img_size
		self.train_size = train_size
		self.val_size = val_size

		# TODO: Add the link to Stanford's site to download the zip if not downloaded
		# TODO: Unzip and set Wordnet and word labels

	def train_val_gen(train_target=64, train_batch=64, val_target=64, val_batch=64):
		val_data = pd.read_csv(VAL_ANNOT , sep='\t', names=['File', 'Class', 'X', 'Y', 'H', 'W'])
		val_data.drop(['X','Y','H', 'W'], axis=1, inplace=True)

		train_datagen = ImageDataGenerator(
		        rescale=1./255,
		        rotation_range=18, # Rotation Angle
		        zoom_range=0.15,  # Zoom Range
		        width_shift_range=0.2, # Width Shift
		        height_shift_range=0.2, # Height Shift
		        shear_range=0.15,  # Shear Range
		        horizontal_flip=True, # Horizontal Flip
		        fill_mode="reflect", # Fills empty with reflections
		        brightness_range=[0.4, 1.6],  # Increasing/decreasing brightness
		)

		train_generator = train_datagen.flow_from_directory(
		        TRAIN,
		        target_size=(train_target, train_target),
		        batch_size=train_batch,
		        class_mode='categorical')

		val_datagen = ImageDataGenerator(rescale=1./255)

		val_generator = val_datagen.flow_from_dataframe(
		    val_data, directory=VAL, 
		    x_col='File', 
		    y_col='Class', 
		    target_size=(val_target, val_target),
		    color_mode='rgb', 
		    class_mode='categorical', 
		    batch_size=val_batch, 
		    shuffle=False, 
		    seed=42
		)

		return train_datagen, val_datagen