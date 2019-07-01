# StackGAN
### Text to Photo-Realistic Image Synthesis
---
#### Architecture
- Stage 1
	- Text Encoder Network
	- Conditional Augmentation Network
	- Generator Network
	- Discriminator Network
	- Adversarial Model
#
- Stage 2
	- Generator Network
	- Discriminator Network
---
#### Checklist
- `tf.keras` with `tf.GradientTape()` training
- XLA Operations
- Distributed training (Mirrored Strategy)
- Evaluation and Tensorboard usage
- Pre-Trained Resnet
- CUB Dataset (11,788 images with 200 classes) with `TFRecords` and `tf.data.Dataset`
- Custom loss functions
- Export model ~ No Subclassing