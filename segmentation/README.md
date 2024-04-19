# Segmentation

## MMSeg version
The `configs` and `tools` are copied from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) `version 1.2.2`.

Other MM package versions:
```shell
mmsegmentation==3.3.0
mmengine==0.10.3
mmcv==2.1.0
```


## Train

The training script is the same as the original mmsegmentation, you may see the official documentation for lauching the training.

|Model|Backbone|Det AP|Seg AP|config/ckpt/log|
|:--:|:--:|:--:|:--:|:--:|
|UperNet|LocalVim-T|||[config](configs/local_mamba/upernet_local_vim-tiny_8xb2-160k_ade20k-512x512.py)|
|UperNet|LocalVim-S|||[config](configs/local_mamba/upernet_local_vim-small_8xb2-160k_ade20k-512x512.py)|
|UperNet|LocalVMamba-T|||[config](configs/local_mamba/upernet_local_vssm_tiny_8xb2-160k_ade20k-512x512.py)|
|UperNet|LocalVMamba-S|||[config](configs/local_mamba/upernet_local_vssm_small_8xb2-160k_ade20k-512x512.py)|


## FLOPs

The FLOPs and #parameters can be measured through the following command:
```shell
python tools/analysis_tools/mamba_get_flops.py <config-file>
```

