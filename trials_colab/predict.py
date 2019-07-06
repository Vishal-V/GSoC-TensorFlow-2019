from argparse import ArgumentParser
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import colorsys
import numpy as np
import random
from config import Config
import model as mask_rcnn
import visualize
import utils
import os
import cv2

ap = ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Weights Directory")
ap.add_argument("-l", "--labels", required=True, help="labels")
ap.add_argument("-i", "--image", required=True, help="Images")
args = vars(ap.parse_args())

LABELS = open(args["labels"]).read().strip().split("\n")
file_name = "images/intersection.jpg"

class MaskConfig(Config):
	NAME = "COCO Trial 2"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = len(LABELS)

configs = MaskConfig()

img = Image.open(args["image"]).convert("RGBA")
img.resize((512, 512))
hsv = [(i / len(LABELS), 1, 1.0) for i in range(len(LABELS))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

print("Loading model weights......")
model = mask_rcnn.MaskRCNN(mode="inference", config=configs, model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

print("Making inferences with Mask R-CNN......")
image = cv2.imread(args["image"])
val = model.detect([image], verbose=1)[0]


for i in range(0, val["rois"].shape[0]):
	classID = val["class_ids"][i]
	mask = val["masks"][:, :, i]
	color = COLORS[classID][::-1]
	img = visualize.apply_mask(img, mask, color, alpha=0.5)

for i in range(0, len(val["scores"])):
	(startY, startX, endY, endX) = val["rois"][i]
	classID = val["class_ids"][i]
	label = LABELS[classID]
	score = val["scores"][i]
	color = [int(c) for c in np.array(COLORS[classID]) * 255]

	draw = ImageDraw.Draw(img)
	draw.rectangle(((startX, startY), (endX, endY)), fill="black")
	draw.text((20, 70), f'{label}, {score}', font=ImageFont.truetype("font_path123"))

	img.save("/content/", "JPEG")
	