# Disentangled Non-Local Neural Networks

## Introduction

[ALGORITHM]

This example is to reproduce ["Disentangled Non-Local Neural Networks"](https://arxiv.org/abs/2006.06668) for semantic segmentation. It is still in progress.

## Citation

```latex
@misc{yin2020disentangled,
    title={Disentangled Non-Local Neural Networks},
    author={Minghao Yin and Zhuliang Yao and Yue Cao and Xiu Li and Zheng Zhang and Stephen Lin and Han Hu},
    year={2020},
    booktitle={ECCV}
}
```

## Results and models (in progress)

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                    download                                                                                                                                                                                    |
|--------|----------|-----------|--------:|---------:|----------------|------:|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dnl    | R-50-D8  | 512x1024  |   40000 |     7.3  | 2.56           | 78.61 | -             | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_40k_cityscapes/dnl_r50-d8_512x1024_40k_cityscapes_20200904_233629-53d4ea93.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_40k_cityscapes/dnl_r50-d8_512x1024_40k_cityscapes-20200904_233629.log.json)     |
| dnl    | R-101-D8 | 512x1024  |   40000 |     10.9 | 1.96           | 78.31 | -             | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_40k_cityscapes/dnl_r101-d8_512x1024_40k_cityscapes_20200904_233629-9928ffef.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_40k_cityscapes/dnl_r101-d8_512x1024_40k_cityscapes-20200904_233629.log.json) |
| dnl    | R-50-D8  | 769x769   |   40000 |     9.2  | 1.50           | 78.44 | 80.27         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_40k_cityscapes/dnl_r50-d8_769x769_40k_cityscapes_20200820_232206-0f283785.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_40k_cityscapes/dnl_r50-d8_769x769_40k_cityscapes-20200820_232206.log.json)         |
| dnl    | R-101-D8 | 769x769   |   40000 |     12.6 | 1.02           | 76.39 | 77.77         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_40k_cityscapes/dnl_r101-d8_769x769_40k_cityscapes_20200820_171256-76c596df.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_40k_cityscapes/dnl_r101-d8_769x769_40k_cityscapes-20200820_171256.log.json)     |
| dnl    | R-50-D8  | 512x1024  |   80000 |     -    | -              | 79.33 | -             | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_80k_cityscapes/dnl_r50-d8_512x1024_80k_cityscapes_20200904_233629-58b2f778.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_80k_cityscapes/dnl_r50-d8_512x1024_80k_cityscapes-20200904_233629.log.json)     |
| dnl    | R-101-D8 | 512x1024  |   80000 |     -    | -              | 80.41 | -             | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_80k_cityscapes/dnl_r101-d8_512x1024_80k_cityscapes_20200904_233629-758e2dd4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_80k_cityscapes/dnl_r101-d8_512x1024_80k_cityscapes-20200904_233629.log.json) |
| dnl    | R-50-D8  | 769x769   |   80000 |     -    | -              | 79.36 | 80.70         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_80k_cityscapes/dnl_r50-d8_769x769_80k_cityscapes_20200820_011925-366bc4c7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_80k_cityscapes/dnl_r50-d8_769x769_80k_cityscapes-20200820_011925.log.json)         |
| dnl    | R-101-D8 | 769x769   |   80000 |     -    | -              | 79.41 | 80.68         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_80k_cityscapes/dnl_r101-d8_769x769_80k_cityscapes_20200821_051111-95ff84ab.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_80k_cityscapes/dnl_r101-d8_769x769_80k_cityscapes-20200821_051111.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                            download                                                                                                                                                                            |
|--------|----------|-----------|--------:|---------:|----------------|------:|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DNL    | R-50-D8  | 512x512   |   80000 |      8.8 | 20.66          | 41.76 | 42.99         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_80k_ade20k/dnl_r50-d8_512x512_80k_ade20k_20200826_183354-1cf6e0c1.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_80k_ade20k/dnl_r50-d8_512x512_80k_ade20k-20200826_183354.log.json)         |
| DNL    | R-101-D8 | 512x512   |   80000 |     12.8 | 12.54          | 43.76 | 44.91         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_80k_ade20k/dnl_r101-d8_512x512_80k_ade20k_20200826_183354-d820d6ea.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_80k_ade20k/dnl_r101-d8_512x512_80k_ade20k-20200826_183354.log.json)     |
| DNL    | R-50-D8  | 512x512   |  160000 |      -   | -              | 41.87 | 43.01         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_160k_ade20k/dnl_r50-d8_512x512_160k_ade20k_20200826_183350-37837798.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_160k_ade20k/dnl_r50-d8_512x512_160k_ade20k-20200826_183350.log.json)     |
| DNL    | R-101-D8 | 512x512   |  160000 |      -   | -              | 44.25 | 45.78         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_160k_ade20k/dnl_r101-d8_512x512_160k_ade20k_20200826_183350-ed522c61.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_160k_ade20k/dnl_r101-d8_512x512_160k_ade20k-20200826_183350.log.json) |
