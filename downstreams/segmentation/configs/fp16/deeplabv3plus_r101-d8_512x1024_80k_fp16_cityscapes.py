_base_ = '../deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
