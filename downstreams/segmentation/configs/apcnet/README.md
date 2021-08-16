# Adaptive Pyramid Context Network for Semantic Segmentation

## Introduction

[ALGORITHM]

```latex
@InProceedings{He_2019_CVPR,
author = {He, Junjun and Deng, Zhongying and Zhou, Lei and Wang, Yali and Qiao, Yu},
title = {Adaptive Pyramid Context Network for Semantic Segmentation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                 download                                                                                                                                                                                                 |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| APCNet | R-50-D8  | 512x1024  |   40000 |      7.7 |           3.57 | 78.02 |         79.26 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes/apcnet_r50-d8_512x1024_40k_cityscapes_20201214_115717-5e88fa33.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes/apcnet_r50-d8_512x1024_40k_cityscapes-20201214_115717.log.json)     |
| APCNet | R-101-D8 | 512x1024  |   40000 |      11.2 |           2.15 | 79.08 |         80.34 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_40k_cityscapes/apcnet_r101-d8_512x1024_40k_cityscapes_20201214_115716-abc9d111.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_40k_cityscapes/apcnet_r101-d8_512x1024_40k_cityscapes-20201214_115716.log.json) |
| APCNet | R-50-D8  | 769x769   |   40000 |      8.7 |           1.52 | 77.89 |         79.75 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_40k_cityscapes/apcnet_r50-d8_769x769_40k_cityscapes_20201214_115717-2a2628d7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_40k_cityscapes/apcnet_r50-d8_769x769_40k_cityscapes-20201214_115717.log.json)         |
| APCNet | R-101-D8 | 769x769   |   40000 |     12.7 |           1.03 | 77.96 |         79.24 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_40k_cityscapes/apcnet_r101-d8_769x769_40k_cityscapes_20201214_115718-b650de90.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_40k_cityscapes/apcnet_r101-d8_769x769_40k_cityscapes-20201214_115718.log.json)     |
| APCNet | R-50-D8  | 512x1024  |   80000 | -        | -              | 78.96 |         79.94 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_80k_cityscapes/apcnet_r50-d8_512x1024_80k_cityscapes_20201214_115716-987f51e3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_80k_cityscapes/apcnet_r50-d8_512x1024_80k_cityscapes-20201214_115716.log.json)     |
| APCNet | R-101-D8 | 512x1024  |   80000 | -        | -              | 79.64 |         80.61 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_80k_cityscapes/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_80k_cityscapes/apcnet_r101-d8_512x1024_80k_cityscapes-20201214_115705.log.json) |
| APCNet | R-50-D8  | 769x769   |   80000 | -        | -              | 78.79 |         80.35 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_80k_cityscapes/apcnet_r50-d8_769x769_80k_cityscapes_20201214_115718-7ea9fa12.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_80k_cityscapes/apcnet_r50-d8_769x769_80k_cityscapes-20201214_115718.log.json)         |
| APCNet | R-101-D8 | 769x769   |   80000 | -        | -              | 78.45 |         79.91 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_80k_cityscapes/apcnet_r101-d8_769x769_80k_cityscapes_20201214_115716-a7fbc2ab.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_80k_cityscapes/apcnet_r101-d8_769x769_80k_cityscapes-20201214_115716.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| APCNet | R-50-D8  | 512x512   |   80000 |      10.1 |          19.61 | 42.20 |         43.30 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_80k_ade20k/apcnet_r50-d8_512x512_80k_ade20k_20201214_115705-a8626293.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_80k_ade20k/apcnet_r50-d8_512x512_80k_ade20k-20201214_115705.log.json)         |
| APCNet | R-101-D8 | 512x512   |   80000 |       13.6 |          13.10 | 45.54 |         46.65 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_80k_ade20k/apcnet_r101-d8_512x512_80k_ade20k_20201214_115704-c656c3fb.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_80k_ade20k/apcnet_r101-d8_512x512_80k_ade20k-20201214_115704.log.json)     |
| APCNet | R-50-D8  | 512x512   |  160000 | -        | -              | 43.40 |         43.94 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_160k_ade20k/apcnet_r50-d8_512x512_160k_ade20k_20201214_115706-25fb92c2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_160k_ade20k/apcnet_r50-d8_512x512_160k_ade20k-20201214_115706.log.json)     |
| APCNet | R-101-D8 | 512x512   |  160000 | -        | -              | 45.41 |         46.63 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_160k_ade20k/apcnet_r101-d8_512x512_160k_ade20k_20201214_115705-73f9a8d7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_160k_ade20k/apcnet_r101-d8_512x512_160k_ade20k-20201214_115705.log.json) |
