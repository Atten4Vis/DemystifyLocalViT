# ResNeSt: Split-Attention Networks

## Introduction

[ALGORITHM]

```latex
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```

## Results and models

### Cityscapes

|   Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                         download                                                                                                                                                                                                         |
|------------|----------|-----------|--------:|---------:|----------------|------:|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCN        | S-101-D8 | 512x1024  |   80000 |     11.4 | 2.39           | 77.56 | 78.98         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x1024_80k_cityscapes/fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x1024_80k_cityscapes/fcn_s101-d8_512x1024_80k_cityscapes-20200807_140631.log.json)                                         |
| PSPNet     | S-101-D8 | 512x1024  |   80000 |     11.8 | 2.52           | 78.57 | 79.19         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x1024_80k_cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x1024_80k_cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes-20200807_140631.log.json)                             |
| DeepLabV3  | S-101-D8 | 512x1024  |   80000 |     11.9 | 1.88           | 79.67 | 80.51         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes_20200807_144429-b73c4270.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes-20200807_144429.log.json)                 |
| DeepLabV3+ | S-101-D8 | 512x1024  |   80000 |     13.2 | 2.36           | 79.62 | 80.27         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x1024_80k_cityscapes/deeplabv3plus_s101-d8_512x1024_80k_cityscapes_20200807_144429-1239eb43.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x1024_80k_cityscapes/deeplabv3plus_s101-d8_512x1024_80k_cityscapes-20200807_144429.log.json) |

### ADE20k

|   Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                 download                                                                                                                                                                                                 |
|------------|----------|-----------|--------:|---------:|----------------|------:|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCN        | S-101-D8 | 512x512   |  160000 |     14.2 | 12.86          | 45.62 | 46.16         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x512_160k_ade20k/fcn_s101-d8_512x512_160k_ade20k_20200807_145416-d3160329.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x512_160k_ade20k/fcn_s101-d8_512x512_160k_ade20k-20200807_145416.log.json)                                         |
| PSPNet     | S-101-D8 | 512x512   |  160000 |     14.2 | 13.02          | 45.44 | 46.28         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x512_160k_ade20k/pspnet_s101-d8_512x512_160k_ade20k_20200807_145416-a6daa92a.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x512_160k_ade20k/pspnet_s101-d8_512x512_160k_ade20k-20200807_145416.log.json)                             |
| DeepLabV3  | S-101-D8 | 512x512   |  160000 |     14.6 | 9.28           | 45.71 | 46.59         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x512_160k_ade20k/deeplabv3_s101-d8_512x512_160k_ade20k_20200807_144503-17ecabe5.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x512_160k_ade20k/deeplabv3_s101-d8_512x512_160k_ade20k-20200807_144503.log.json)                 |
| DeepLabV3+ | S-101-D8 | 512x512   |  160000 |     16.2 | 11.96          | 46.47 | 47.27         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x512_160k_ade20k/deeplabv3plus_s101-d8_512x512_160k_ade20k_20200807_144503-27b26226.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x512_160k_ade20k/deeplabv3plus_s101-d8_512x512_160k_ade20k-20200807_144503.log.json) |
