# Runner Code
# TODO: Train Stage 1 
# TODO: Train Stage 2

import model as stack_gan

stage1 = stack_gan.StackGanStage1()
stage2 = stack_gan.StackGanStage2()

stage1.train_stage1()
stage2.train_stage2()
