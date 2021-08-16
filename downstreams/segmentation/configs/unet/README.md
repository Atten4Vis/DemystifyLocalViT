# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Introduction

[ALGORITHM]

```latex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

## Results and models

### DRIVE

| Backbone | Head | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) | Dice  |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UNet-S5-D16 | FCN  |   584x565 |      64x64 |          42x42 | 40000 |         0.680 |  - | 78.67 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_64x64_40k_drive/fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-26cee593.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_64x64_40k_drive/fcn_unet_s5-d16_64x64_40k_drive-20201223_191051.log.json)         |
| UNet-S5-D16 | PSPNet  |   584x565 |      64x64 |          42x42 | 40000 |         0.599 |  - | 78.62 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_64x64_40k_drive/pspnet_unet_s5-d16_64x64_40k_drive_20201227_181818-aac73387.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_64x64_40k_drive/pspnet_unet_s5-d16_64x64_40k_drive-20201227_181818.log.json)         |
| UNet-S5-D16 | DeepLabV3  |   584x565 |      64x64 |          42x42 | 40000 |         0.596 |  - | 78.69 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_64x64_40k_drive/deeplabv3_unet_s5-d16_64x64_40k_drive_20201226_094047-0671ff20.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_64x64_40k_drive/deeplabv3_unet_s5-d16_64x64_40k_drive-20201226_094047.log.json)         |

### STARE

| Backbone | Head | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) | Dice  |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UNet-S5-D16 | FCN  |   605x700 |      128x128 |          85x85 | 40000 |         0.968 |  - | 81.02 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_128x128_40k_stare/fcn_unet_s5-d16_128x128_40k_stare_20201223_191051-6ea7cfda.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_128x128_40k_stare/fcn_unet_s5-d16_128x128_40k_stare-20201223_191051.log.json)         |
| UNet-S5-D16 | PSPNet  |   605x700 |      128x128 |          85x85 | 40000 |         0.982 |  - | 81.22 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_stare/pspnet_unet_s5-d16_128x128_40k_stare_20201227_181818-3c2923c4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_stare/pspnet_unet_s5-d16_128x128_40k_stare-20201227_181818.log.json)         |
| UNet-S5-D16 | DeepLabV3  |   605x700 |      128x128 |          85x85 | 40000 |         0.999 |  - | 80.93 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_stare/deeplabv3_unet_s5-d16_128x128_40k_stare_20201226_094047-93dcb93c.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_stare/deeplabv3_unet_s5-d16_128x128_40k_stare-20201226_094047.log.json)         |

### CHASE_DB1

| Backbone | Head | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) | Dice  |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UNet-S5-D16 | FCN  |   960x999 |      128x128 |          85x85 | 40000 |         0.968 |  - | 80.24 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_128x128_40k_chase_db1/fcn_unet_s5-d16_128x128_40k_chase_db1_20201223_191051-95852f45.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_128x128_40k_chase_db1/fcn_unet_s5-d16_128x128_40k_chase_db1-20201223_191051.log.json)         |
| UNet-S5-D16 | PSPNet  |   960x999 |      128x128 |          85x85 | 40000 |         0.982 |  - | 80.36 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1/pspnet_unet_s5-d16_128x128_40k_chase_db1_20201227_181818-68d4e609.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1/pspnet_unet_s5-d16_128x128_40k_chase_db1-20201227_181818.log.json)         |
| UNet-S5-D16 | DeepLabV3  |   960x999 |      128x128 |          85x85 | 40000 |         0.999 |  - | 80.47 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1/deeplabv3_unet_s5-d16_128x128_40k_chase_db1_20201226_094047-4c5aefa3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1/deeplabv3_unet_s5-d16_128x128_40k_chase_db1-20201226_094047.log.json)         |

### HRF

| Backbone | Head | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) | Dice  |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UNet-S5-D16 | FCN  |   2336x3504 |      256x256 |          170x170 | 40000 |         2.525 |  - | 79.45 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_256x256_40k_hrf/fcn_unet_s5-d16_256x256_40k_hrf_20201223_173724-df3ec8c4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_256x256_40k_hrf/fcn_unet_s5-d16_256x256_40k_hrf-20201223_173724.log.json)         |
| UNet-S5-D16 | PSPNet  |   2336x3504 |      256x256 |          170x170 | 40000 |         2.588 |  - | 80.07 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_256x256_40k_hrf/pspnet_unet_s5-d16_256x256_40k_hrf_20201227_181818-fdb7e29b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_256x256_40k_hrf/pspnet_unet_s5-d16_256x256_40k_hrf-20201227_181818.log.json)         |
| UNet-S5-D16 | DeepLabV3  |   2336x3504 |      256x256 |          170x170 | 40000 |         2.604 |  - | 80.21 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf/deeplabv3_unet_s5-d16_256x256_40k_hrf_20201226_094047-3a1fdf85.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf/deeplabv3_unet_s5-d16_256x256_40k_hrf-20201226_094047.log.json)         |
