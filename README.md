# GSoC-Progress-Blog
All my commits to Tensorflow during GSoC and a growing log to track my progress
<br>
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
