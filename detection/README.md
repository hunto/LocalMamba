# Detection

## MMDet version
The `configs` and `tools` are copied from [mmdetection](https://github.com/open-mmlab/mmdetection) `version 3.3.0`.

Other MM package versions:
```shell
mmsegmentation==1.2.2
mmengine==0.10.3
mmcv==2.1.0
```


## Train

The training script is the same as the original mmdetection, you may see the official documentation for lauching the training.

|Model|Backbone|Det AP|Seg AP|config/ckpt/log|
|:--:|:--:|:--:|:--:|:--:|
|Cascade Mask R-CNN|LocalVim-T|||[config](configs/local_mamba/vitdet_cascade_mask-rcnn_local_vim_tiny_lsj-100e.py)|
|Cascade Mask R-CNN|LocalVim-S|||[config](configs/local_mamba/vitdet_cascade_mask-rcnn_local_vim_small_lsj-100e.py)|


|Model|Schedule|Backbone|Det AP|Seg AP|config/ckpt/log|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Mask R-CNN|1x|LocalVMamba-T|||[config](configs/local_mamba/mask_rcnn_local_vssm_tiny_fpn_coco.py)|
|Mask R-CNN|1x|LocalVMamba-S|||[config](configs/local_mamba/mask_rcnn_local_vssm_small_fpn_coco.py)|
|Mask R-CNN|MS 3x|LocalVMamba-T|||[config](configs/local_mamba/mask_rcnn_local_vssm_tiny_fpn_coco_ms_3x.py)|
|Mask R-CNN|MS 3x|LocalVMamba-S|||[config](configs/local_mamba/mask_rcnn_local_vssm_small_fpn_coco_ms_3x.py)|



## FLOPs

The FLOPs and #parameters can be measured through the following command:
```shell
python tools/analysis_tools/mamba_get_flops.py <config-file>
```

