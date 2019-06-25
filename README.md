# GSoC
---
Repository for Google Summer of Code 2019
---------------------------------------------
|Evaluation|Task|Link|Status|Pull Request|
|---|---|---|---|---|
|E1|Autoencoder Migration |[Here](https://github.com/Vishal-V/GSoC/tree/master/autoencoder)| Complete |[ #68](https://github.com/tensorflow/examples/pull/68)
|E1|Boosted Trees Migration|[Here](https://github.com/Vishal-V/GSoC/tree/master/boosted_trees)|  WIP - Minor Bug Fixes |-|
|E1|Mask R-CNN Migration|[Here](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn)|<ul><li>Part Migrated</li><li>Working on Testing and Migration Guide</li></ul>|-|
|E1|Custom ResNet with TinyImageNet - Part 1 |[Here](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb)|<ul><li>Model Demo Ready</li><li>Working on CLR, Cutout/Occlusion and Progressive Resizing</li></ul> |-|
|E1|Hyperparameter Search|[Here](https://github.com/Vishal-V/GSoC/blob/master/keras_tuner/hyperparamter_search.ipynb)|  WIP - Keras Tuner, tf.keras wrappers |-|
#
# GSoC-Progress-Blog
---
### May 11: Updated progress with mentor Amit Sabne
- RetinaNet (Was last updated on July 13, 2018) - Medium blogs, Focal Loss paper by Yi-Lin et al, Alpha weighted Focal Loss
- GANs - The TFGAN repo has a few varieties of GANs but they were updated 7-11 months ago. So, I looked into and worked on SRGANs, StackGANs, CycleGANs to add a folder for them
- Training strategies from benchmarks.ai and Jeremy Howard's blog including SuperConvergence, Progressive Resizing, Exponential CLR, LR decay...etc. 
- Adding tf.data pipelines to load images apart from using flow_from_directory with keras' `ImageDataGenerator` class
- Will be looking into the current implementations of wide_deep and boosted trees to recreate them. (Maybe towards the end of next month)

### May 12: Google I/O Tensoflow changes and KerasTuner
- Adding `tf.distribute.startegy` or `tf.distribute.MirroredStrategy` for scoping `tf.keras` models for distributed leaning and adding `tf.function` decorators wherever required to optimise the execution.
- Subclassing the models and layers is considered best practice. So, will change some of them from The Functional API to subclassing API
- Applied for beta access to use KeasTuner to tune parameters quickly and effectively
  
### June 21: Some Proposed Additions
- Keras-Tuner Tutorial : Based off of a trending Reddit thread
- Tiny ImageNet Advanced CNNs : With a suitable migration guide
- Tensorflow 2.0 Mask R-CNN
- GAN architectures
