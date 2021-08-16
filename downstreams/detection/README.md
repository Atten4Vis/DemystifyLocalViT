# Depth-wise convolution based network for Object Detection

This is the detection code for paper [demystifying local vision transformer](https://arxiv.org/pdf/2106.04263.pdf). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DWNet-T | ImageNet-1K | 3x | 49.9 | 43.4 | 82M | 730G | [config](configs/dwnet/cascade_mask_rcnn_dwnet_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/Atten4Vis/DemystifyLocalViT/releases/download/prerelease/cascade_mask_rcnn_dwnet_tiny.pth)|
| DWNet-B | ImageNet-1K | 3x | 51.0 | 44.1 | 132M | 924G | [config](configs/dwnet/cascade_mask_rcnn_dwnet_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/Atten4Vis/DemystifyLocalViT/releases/download/prerelease/cascade_mask_rcnn_dwnet_base.pth)|
| Dynamic-DWNet-T | ImageNet-1K | 3x | 50.5 | 43.7 | 108M | 730G | [config](configs/dwnet/cascade_mask_rcnn_dynamic_dwnet_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/Atten4Vis/DemystifyLocalViT/releases/download/prerelease/cascade_mask_rcnn_dynamic_dwnet_tiny.pth)|
| Dynamic-DWNet-B | ImageNet-1K | 3x | 51.2 | 44.4 | 219M | 924G | [config](configs/dwnet/cascade_mask_rcnn_dynamic_dwnet_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/Atten4Vis/DemystifyLocalViT/releases/download/prerelease/cascade_mask_rcnn_dynamic_dwnet_base.pth)|

**Notes**: 

- **Pre-trained models can be downloaded from [ImageNet Classification](https://github.com/Atten4Vis/DemystifyLocalViT)**.



## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```

### Apex (optional):
Borrow from Swin Transformer, 
we use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```
