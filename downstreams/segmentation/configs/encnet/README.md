# Context Encoding for Semantic Segmentation

## Introduction

[ALGORITHM]

```latex
@InProceedings{Zhang_2018_CVPR,
author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
title = {Context Encoding for Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                 download                                                                                                                                                                                                 |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| encnet | R-50-D8  | 512x1024  |   40000 |      8.6 |           4.58 | 75.67 |         77.08 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_40k_cityscapes/encnet_r50-d8_512x1024_40k_cityscapes_20200621_220958-68638a47.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_40k_cityscapes/encnet_r50-d8_512x1024_40k_cityscapes-20200621_220958.log.json)     |
| encnet | R-101-D8 | 512x1024  |   40000 |     12.1 |           2.66 | 75.81 |         77.21 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_40k_cityscapes/encnet_r101-d8_512x1024_40k_cityscapes_20200621_220933-35e0a3e8.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_40k_cityscapes/encnet_r101-d8_512x1024_40k_cityscapes-20200621_220933.log.json) |
| encnet | R-50-D8  | 769x769   |   40000 |      9.8 |           1.82 | 76.24 |         77.85 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_40k_cityscapes/encnet_r50-d8_769x769_40k_cityscapes_20200621_220958-3bcd2884.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_40k_cityscapes/encnet_r50-d8_769x769_40k_cityscapes-20200621_220958.log.json)         |
| encnet | R-101-D8 | 769x769   |   40000 |     13.7 |           1.26 | 74.25 |         76.25 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_40k_cityscapes/encnet_r101-d8_769x769_40k_cityscapes_20200621_220933-2fafed55.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_40k_cityscapes/encnet_r101-d8_769x769_40k_cityscapes-20200621_220933.log.json)     |
| encnet | R-50-D8  | 512x1024  |   80000 | -        | -              | 77.94 |         79.13 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_80k_cityscapes/encnet_r50-d8_512x1024_80k_cityscapes_20200622_003554-fc5c5624.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_80k_cityscapes/encnet_r50-d8_512x1024_80k_cityscapes-20200622_003554.log.json)     |
| encnet | R-101-D8 | 512x1024  |   80000 | -        | -              | 78.55 |         79.47 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_80k_cityscapes/encnet_r101-d8_512x1024_80k_cityscapes_20200622_003555-1de64bec.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_80k_cityscapes/encnet_r101-d8_512x1024_80k_cityscapes-20200622_003555.log.json) |
| encnet | R-50-D8  | 769x769   |   80000 | -        | -              | 77.44 |         78.72 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_80k_cityscapes/encnet_r50-d8_769x769_80k_cityscapes_20200622_003554-55096dcb.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_80k_cityscapes/encnet_r50-d8_769x769_80k_cityscapes-20200622_003554.log.json)         |
| encnet | R-101-D8 | 769x769   |   80000 | -        | -              | 76.10 |         76.97 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_80k_cityscapes/encnet_r101-d8_769x769_80k_cityscapes_20200622_003555-470ef79d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_80k_cityscapes/encnet_r101-d8_769x769_80k_cityscapes-20200622_003555.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| encnet | R-50-D8  | 512x512   |   80000 |     10.1 |          22.81 | 39.53 |         41.17 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_80k_ade20k/encnet_r50-d8_512x512_80k_ade20k_20200622_042412-44b46b04.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_80k_ade20k/encnet_r50-d8_512x512_80k_ade20k-20200622_042412.log.json)         |
| encnet | R-101-D8 | 512x512   |   80000 |     13.6 |          14.87 | 42.11 |         43.61 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_80k_ade20k/encnet_r101-d8_512x512_80k_ade20k_20200622_101128-dd35e237.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_80k_ade20k/encnet_r101-d8_512x512_80k_ade20k-20200622_101128.log.json)     |
| encnet | R-50-D8  | 512x512   |  160000 | -        | -              | 40.10 |         41.71 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_160k_ade20k/encnet_r50-d8_512x512_160k_ade20k_20200622_101059-b2db95e0.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_160k_ade20k/encnet_r50-d8_512x512_160k_ade20k-20200622_101059.log.json)     |
| encnet | R-101-D8 | 512x512   |  160000 | -        | -              | 42.61 |         44.01 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_160k_ade20k/encnet_r101-d8_512x512_160k_ade20k_20200622_073348-7989641f.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_160k_ade20k/encnet_r101-d8_512x512_160k_ade20k-20200622_073348.log.json) |
