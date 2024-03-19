from . import operations
from . import operations_resnet 

# models which use timm's registry
try:
    import timm
    _has_timm = True
except ModuleNotFoundError:
    _has_timm = False

if _has_timm:
    from . import lightvit
