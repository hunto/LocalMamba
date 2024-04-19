import os
from torch import nn

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "lib.models.local_vim", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)

Backbone_LocalVim: nn.Module = build.Backbone_LocalVisionMamba

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_LocalVim(BaseModule, Backbone_LocalVim):
    def __init__(self, *args, **kwargs):
        Backbone_LocalVim.__init__(self, *args, **kwargs)
        self._is_init = True
        self.init_cfg = None


build = import_abspy(
    "lib.models.local_vmamba", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)

Backbone_LocalVSSM: nn.Module = build.Backbone_LocalVSSM
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_LocalVSSM(BaseModule, Backbone_LocalVSSM):
    def __init__(self, *args, **kwargs):
        Backbone_LocalVSSM.__init__(self, *args, **kwargs)
        self._is_init = True
        self.init_cfg = None
