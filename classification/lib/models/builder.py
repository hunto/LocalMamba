import yaml
import torch
import torchvision
import logging

from .nas_model import gen_nas_model
from .darts_model import gen_darts_model
from .mobilenet_v1 import MobileNetV1
from . import resnet


logger = logging.getLogger()


def build_model(args, model_name, pretrained=False, pretrained_ckpt=''):
    if model_name.lower() == 'nas_model':
        # model with architectures specific in yaml file
        model = gen_nas_model(yaml.safe_load(open(args.model_config, 'r')), drop_rate=args.drop, 
                              drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)

    elif model_name.lower() == 'darts_model':
        # DARTS evaluation models
        model = gen_darts_model(yaml.safe_load(open(args.model_config, 'r')), args.dataset, drop_rate=args.drop, 
                                drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)

    elif model_name.lower() == 'nas_pruning_model':
        # model with architectures specific in yaml file
        # the model is searched by pruning algorithms
        from edgenn.models import EdgeNNModel
        model_config = yaml.safe_load(open(args.model_config, 'r'))
        channel_settings = model_config.pop('channel_settings')
        model = gen_nas_model(model_config, drop_rate=args.drop, drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)
        edgenn_model = EdgeNNModel(model, loss_fn=None, pruning=True, input_shape=args.input_shape)
        logger.info(edgenn_model.graph)
        edgenn_model.fold_dynamic_nn(channel_settings['choices'], channel_settings['bins'], channel_settings['min_bins'])
        logger.info(model)

    elif model_name.lower().startswith('resnet'):
        # resnet variants (the same as torchvision)
        model = getattr(resnet, model_name.lower())(num_classes=args.num_classes)

    elif model_name.lower() == 'mobilenet_v1':
        # mobilenet v1
        model = MobileNetV1(num_classes=args.num_classes)

    elif model_name.startswith('tv_'):
        # build model using torchvision
        import torchvision
        model = getattr(torchvision.models, model_name[3:])(pretrained=pretrained)

    elif model_name.startswith('timm_'):
        # build model using timm
        # we import local_vim and local_vmamba here to register the models
        if 'local_vim' in model_name:
            from lib.models import local_vim
        if 'local_vssm' in model_name:
            from lib.models import local_vmamba
        import timm
        model = timm.create_model(model_name[5:], pretrained=pretrained, drop_path_rate=args.drop_path_rate)

    elif model_name.startswith('cifar_'):
        from .cifar import model_dict
        model_name = model_name[6:]
        model = model_dict[model_name](num_classes=args.num_classes)
    else:
        raise RuntimeError(f'Model {model_name} not found.')

    if pretrained and pretrained_ckpt != '':
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = \
                model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) != 0:
            logger.info(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.info(f'Unexpected keys in source state dict: {unexpected_keys}')

    return model


def build_edgenn_model(args, edgenn_cfgs=None):
    import edgenn
    if args.model.lower() in ['nas_model', 'nas_pruning_model']:
        # gen model with yaml config first
        model = gen_nas_model(yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader), drop_rate=args.drop, drop_path_rate=args.drop_path_rate)
        # wrap the model with EdgeNNModel
        model = edgenn.models.EdgeNNModel(model, loss_fn, pruning=(args.model=='nas_pruning_model'))

    elif args.model == 'edgenn':
        # build model from edgenn
        model = edgenn.build_model(edgenn_cfgs.model)

    else:
        raise RuntimeError(f'Model {args.model} not found.')

    return model
