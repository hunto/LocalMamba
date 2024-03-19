# Image Classification SOTA  

`Image Classification SOTA` is an image classification toolbox based on PyTorch.

## Updates  
### May 27, 2022  
* Add knowledge distillation methods (KD and [DIST](https://github.com/hunto/DIST_KD)).  

### March 24, 2022  
* Support training strategies in DeiT (ViT).

### March 11, 2022  
* Release training code.

## Supported Algorithms  
### Structural Re-parameterization (Rep)  
* DBB (CVPR 2021) [[paper]](https://arxiv.org/abs/2103.13425) [[original repo]](https://github.com/DingXiaoH/DiverseBranchBlock)
* DyRep (CVPR 2022) [[README]](https://github.com/hunto/DyRep)

### Knowledge Distillation (KD)  
* KD [[paper]](https://arxiv.org/abs/1503.02531)  
* DIST [[README]](https://github.com/hunto/DIST_KD) [[paper]](https://arxiv.org/abs/2205.10536)  

## Requirements
```
torch>=1.0.1
torchvision
```

## Getting Started  
### Prepare datasets  
It is recommended to symlink the dataset root to `image_classification_sota/data`. Then the file structure should be like  
```
image_classification_sota
├── lib
├── tools
├── configs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
│   │   ├── cifar-100-python
```

### Training configurations  
* `Strategies`: The training strategies are configured using yaml file or arguments. Examples are in `configs/strategies` directory.

### Train a model  

* Training with a single GPU  
    ```shell
    python tools/train.py -c ${CONFIG} --model ${MODEL} [optional arguments]
    ```

* Training with multiple GPUs
    ```shell
    sh tools/dist_train.sh ${GPU_NUM} ${CONFIG} ${MODEL} [optional arguments]
    ```

* For slurm users
    ```shell
    sh tools/slurm_train.sh ${PARTITION} ${GPU_NUM} ${CONFIG} ${MODEL} [optional arguments]
    ```

**Examples**  
* Train ResNet-50 on ImageNet
    ```shell
    sh tools/dist_train.sh 8 configs/strategies/resnet/resnet.yaml resnet50 --experiment imagenet_res50
    ```

* Train MobileNetV2 on ImageNet
    ```shell
    sh tools/dist_train.sh 8 configs/strategies/MBV2/mbv2.yaml nas_model --model-config configs/models/MobileNetV2/MobileNetV2.yaml --experiment imagenet_mbv2
    ```

* Train VGG-16 on CIFAR-10
    ```shell
    sh tools/dist_train.sh 1 configs/strategies/CIFAR/cifar.yaml nas_model --model-config configs/models/VGG/vgg16_cifar10.yaml --experiment cifar10_vgg16
    ```

## Projects based on Image Classification SOTA  
* [CVPR 2022] [DyRep](https://github.com/hunto/DyRep): Bootstrapping Training with Dynamic Re-parameterization
* [NeurIPS 2022] [DIST](https://github.com/hunto/DIST_KD): Knowledge Distillation from A Stronger Teacher
* [LightViT](https://github.com/hunto/LightViT): Towards Light-Weight Convolution-Free Vision Transformers
