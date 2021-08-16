# Asymmetric Non-local Neural Networks for Semantic Segmentation

## Introduction

[ALGORITHM]

```latex
@inproceedings{annn,
  author    = {Zhen Zhu and
               Mengde Xu and
               Song Bai and
               Tengteng Huang and
               Xiang Bai},
  title     = {Asymmetric Non-local Neural Networks for Semantic Segmentation},
  booktitle={International Conference on Computer Vision},
  year      = {2019},
  url       = {http://arxiv.org/abs/1908.07678},
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                        download                                                                                                                                                                                        |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ANN    | R-50-D8  | 512x1024  |   40000 |        6 |           3.71 | 77.40 |         78.57 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x1024_40k_cityscapes/ann_r50-d8_512x1024_40k_cityscapes_20200605_095211-049fc292.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x1024_40k_cityscapes/ann_r50-d8_512x1024_40k_cityscapes_20200605_095211.log.json)     |
| ANN    | R-101-D8 | 512x1024  |   40000 |      9.5 |           2.55 | 76.55 |         78.85 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x1024_40k_cityscapes/ann_r101-d8_512x1024_40k_cityscapes_20200605_095243-adf6eece.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x1024_40k_cityscapes/ann_r101-d8_512x1024_40k_cityscapes_20200605_095243.log.json) |
| ANN    | R-50-D8  | 769x769   |   40000 |      6.8 |           1.70 | 78.89 |         80.46 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_769x769_40k_cityscapes/ann_r50-d8_769x769_40k_cityscapes_20200530_025712-2b46b04d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_769x769_40k_cityscapes/ann_r50-d8_769x769_40k_cityscapes_20200530_025712.log.json)         |
| ANN    | R-101-D8 | 769x769   |   40000 |     10.7 |           1.15 | 79.32 |         80.94 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_769x769_40k_cityscapes/ann_r101-d8_769x769_40k_cityscapes_20200530_025720-059bff28.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_769x769_40k_cityscapes/ann_r101-d8_769x769_40k_cityscapes_20200530_025720.log.json)     |
| ANN    | R-50-D8  | 512x1024  |   80000 | -        | -              | 77.34 |         78.65 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x1024_80k_cityscapes/ann_r50-d8_512x1024_80k_cityscapes_20200607_101911-5a9ad545.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x1024_80k_cityscapes/ann_r50-d8_512x1024_80k_cityscapes_20200607_101911.log.json)     |
| ANN    | R-101-D8 | 512x1024  |   80000 | -        | -              | 77.14 |         78.81 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x1024_80k_cityscapes/ann_r101-d8_512x1024_80k_cityscapes_20200607_013728-aceccc6e.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x1024_80k_cityscapes/ann_r101-d8_512x1024_80k_cityscapes_20200607_013728.log.json) |
| ANN    | R-50-D8  | 769x769   |   80000 | -        | -              | 78.88 |         80.57 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_769x769_80k_cityscapes/ann_r50-d8_769x769_80k_cityscapes_20200607_044426-cc7ff323.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_769x769_80k_cityscapes/ann_r50-d8_769x769_80k_cityscapes_20200607_044426.log.json)         |
| ANN    | R-101-D8 | 769x769   |   80000 | -        | -              | 78.80 |         80.34 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_769x769_80k_cityscapes/ann_r101-d8_769x769_80k_cityscapes_20200607_013713-a9d4be8d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_769x769_80k_cityscapes/ann_r101-d8_769x769_80k_cityscapes_20200607_013713.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                download                                                                                                                                                                                |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ANN    | R-50-D8  | 512x512   |   80000 |      9.1 |          21.01 | 41.01 |         42.30 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_80k_ade20k/ann_r50-d8_512x512_80k_ade20k_20200615_014818-26f75e11.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_80k_ade20k/ann_r50-d8_512x512_80k_ade20k_20200615_014818.log.json)         |
| ANN    | R-101-D8 | 512x512   |   80000 |     12.5 |          14.12 | 42.94 |         44.18 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_80k_ade20k/ann_r101-d8_512x512_80k_ade20k_20200615_014818-c0153543.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_80k_ade20k/ann_r101-d8_512x512_80k_ade20k_20200615_014818.log.json)     |
| ANN    | R-50-D8  | 512x512   |  160000 | -        | -              | 41.74 |         42.62 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_160k_ade20k/ann_r50-d8_512x512_160k_ade20k_20200615_231733-892247bc.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_160k_ade20k/ann_r50-d8_512x512_160k_ade20k_20200615_231733.log.json)     |
| ANN    | R-101-D8 | 512x512   |  160000 | -        | -              | 42.94 |         44.06 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_160k_ade20k/ann_r101-d8_512x512_160k_ade20k_20200615_231733-955eb1ec.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_160k_ade20k/ann_r101-d8_512x512_160k_ade20k_20200615_231733.log.json) |

### Pascal VOC 2012 + Aug

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                  download                                                                                                                                                                                  |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ANN    | R-50-D8  | 512x512   |   20000 |        6 |          20.92 | 74.86 |         76.13 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_20k_voc12aug/ann_r50-d8_512x512_20k_voc12aug_20200617_222246-dfcb1c62.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_20k_voc12aug/ann_r50-d8_512x512_20k_voc12aug_20200617_222246.log.json)     |
| ANN    | R-101-D8 | 512x512   |   20000 |      9.5 |          13.94 | 77.47 |         78.70 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_20k_voc12aug/ann_r101-d8_512x512_20k_voc12aug_20200617_222246-2fad0042.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_20k_voc12aug/ann_r101-d8_512x512_20k_voc12aug_20200617_222246.log.json) |
| ANN    | R-50-D8  | 512x512   |   40000 | -        | -              | 76.56 |         77.51 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_40k_voc12aug/ann_r50-d8_512x512_40k_voc12aug_20200613_231314-b5dac322.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x512_40k_voc12aug/ann_r50-d8_512x512_40k_voc12aug_20200613_231314.log.json)     |
| ANN    | R-101-D8 | 512x512   |   40000 | -        | -              | 76.70 |         78.06 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_40k_voc12aug/ann_r101-d8_512x512_40k_voc12aug_20200613_231314-bd205bbe.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r101-d8_512x512_40k_voc12aug/ann_r101-d8_512x512_40k_voc12aug_20200613_231314.log.json) |
