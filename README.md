<img src="assets/gsoc.png" width="556px" height="112px"/>
  
## Google Summer of Code 2019: **Final Work Product**
### **Organisation**: Tensorflow
### **Mentors**
- Tomer Kaftan ([@tomerk](https://github.com/tomerk))
- Amit Sabne ([@amitsabne1](https://github.com/amitsabne1))
- Vikram Tiwari ([@VikramTiwari](https://github.com/VikramTiwari))
- Katherine Wu ([@k-w-w](https://github.com/k-w-w))
- Paige Bailey ([@dynamicwebpaige](https://github.com/dynamicwebpaige))
## **Aim**
Re-building the official Tensorflow models to make it TF 2.0 compatible. This project proposes holistic improvements to the models repository to upgrade models/research and models/official. The project scope also includes building new deep learning models and features to improve research prototyping with Tensorflow 2.0. Creating model migration guides for the official models will enable onboarding to TF 2.0 with eager mode for R&D and graph mode for serving.

The official models (including Mask R-CNN and StackGAN) will be upgraded with tf.data pipelines and distributed training with DistributionStrategies. Other improvements include bug fixes and the use of tf.GradientTape to compute the gradients more efficiently using Autodifferentiation. The Variational Autoencoders and the GANs projects will be recreated with TF 2.0 with features to train efficiently and export to deploy.
#
## **Progress**
|Evaluation|Task|Link|Status|Pull Request|
|---|---|---|---|---|
|E1|Autoencoder Migration |[Here](https://github.com/Vishal-V/GSoC/tree/master/autoencoder)| Complete |[ #68](https://github.com/tensorflow/examples/pull/68), [ #6795](https://github.com/tensorflow/models/pull/6795)
|E1|Boosted Trees Migration|[Here](https://github.com/Vishal-V/GSoC/tree/master/boosted_trees)|  WIP - Minor Bug Fixes |[Branch](https://github.com/Vishal-V/examples-1/tree/boosted-tree-migration)|
|E1|Hyperparameter Search|[Here](https://github.com/Vishal-V/GSoC/blob/master/keras_tuner/hyperparamter_search.ipynb)| Keras Tuner not yet in Tf 2.0 |[Branch](https://github.com/Vishal-V/examples-1/tree/hyperparam-optimization)|
|E1|Custom ResNet - Part 1 |[Here](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb)|Complete |[ #79](https://github.com/tensorflow/examples/pull/79)|
|E2|StackGAN Model|[Here](https://github.com/Vishal-V/GSoC/tree/master/stack_gan)| Complete |[ #77](https://github.com/tensorflow/examples/pull/77)
|E2|Mask R-CNN Migration|[Here](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn)|Migration Guide|[ #78](https://github.com/tensorflow/examples/pull/78)|
|E2|Custom ResNet - Part 2 |[Here](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb)|Complete|[ #79](https://github.com/tensorflow/examples/pull/79)|
|E3|Face Aging - Model |[Here](https://github.com/Vishal-V/GSoC/blob/master/face_app/model.py)|Complete|[ #]()|
|E3|Custom ResNet - Notebook |[Here](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb)|Complete|[ #79](https://github.com/tensorflow/examples/pull/79)|
|E1|Autoencoder - Notebook |[Here](https://github.com/Vishal-V/GSoC/blob/master/autoencoder/notebook/autoencoder.ipynb)| Complete |[ #68](https://github.com/tensorflow/examples/pull/68)|
|E3|StackGAN - Notebook |[Here](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb)|Ongoing|[ #77](https://github.com/tensorflow/examples/pull/77)|
|E3|Mask R-CNN - TF 2.0 Model |[Here](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn)|Ongoing|[ #78](https://github.com/tensorflow/examples/pull/78)|
#
## **Work Done**
### Autoencoder
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/autoencoder/model
- **Migraton Guide**: https://github.com/Vishal-V/GSoC/blob/master/autoencoder/README.md  
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/GSoC/blob/master/autoencoder/notebook/autoencoder.ipynb), [Colab Link](https://colab.research.google.com/drive/1aZ0mEFEui1A7FPWMylvjiVZ5XZIX7S9w)
### Custom ResNet for TinyImageNet
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/tiny_imagenet_custom_resnet/model
- **Notebook**: [GitHub Link](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb), [Colab Link](https://colab.research.google.com/drive/1SZLecFzKuU7TVCoCzTq285sEbjlT12A6)
### StackGAN
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/stack_gan
- **Notebook**: [GitHub Link](), [Colab Link]()
### Age-cGAN
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/face_app
- **Notebook**: [GitHub Link](), [Colab Link]()
### Hyperparameter Tuning
- **Notebook**: [GitHub Link](), [Colab Link]()
### Mask R-CNN [WIP]
- **Model**: https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn
- **Migraton Guide**: https://github.com/Vishal-V/GSoC/blob/master/mask_rcnn/README.md
- **Notebook**: [GitHub Link](), [Colab Link]()

## **Deliverables**
- [**Autoencoder**](https://github.com/Vishal-V/GSoC/tree/master/autoencoder/model) - [tensorflow/examples #68]()
- [**StackGAN**](https://github.com/Vishal-V/GSoC/tree/master/stack_gan) -  [tensorflow/examples #77]()
- [**StackGAN Notebook**](https://github.com/Vishal-V/GSoC/blob/master/stack_gan/notebook/stack_gan.ipynb) -  [tensorflow/examples #77]()
- [**Age-cGAN**](https://github.com/Vishal-V/GSoC/tree/master/face_app) -  [tensorflow/examples #83]()
- [**FaceApp Notebook**]() - [tensorflow/examples #]() 
- [**Mask R-CNN**](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn) -  [tensorflow/examples #78]()
- [**Mask R-CNN Notebook**](https://github.com/Vishal-V/GSoC/tree/master/mask_rcnn/notebooks) -  [tensorflow/examples #78]()
- [**Hyperparameter Search**](https://github.com/Vishal-V/GSoC/blob/master/keras_tuner/hyperparamter_search.ipynb) -  [tensorflow/examples #84]()
- [**Custom ResNet for TinyImageNet**](https://github.com/Vishal-V/GSoC/tree/master/tiny_imagenet_custom_resnet/model) -  [tensorflow/examples #79]()
- [**Custom ResNet for TinyImageNet Notebook**](https://github.com/Vishal-V/GSoC/blob/master/tiny_imagenet_custom_resnet/tiny_imagenet_custom_resnet.ipynb) -  [tensorflow/examples #79]()
## **Progress**
All pull requests can be found here [PRs Link](https://github.com/tensorflow/examples/pulls/Vishal-V)
- **Phase 1**
- **Phase 2**
- **Phase 3**
## **What's left?**
- **Mask R-CNN**: 
- **Age-cGAN Trained Weights**
- **StackGAN Trained Weights**
## **Challenges**
## **Learnings**