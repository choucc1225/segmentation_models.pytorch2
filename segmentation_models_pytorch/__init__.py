from . import datasets
from . import encoders
from . import decoders
from . import losses
from . import metrics

from .decoders.unet import Unet
from .decoders.unetplusplus import UnetPlusPlus
from .decoders.manet import MAnet
from .decoders.linknet import Linknet
from .decoders.fpn import FPN
from .decoders.pspnet import PSPNet
from .decoders.deeplabv3 import DeepLabV3, DeepLabV3Plus
from .decoders.pan import PAN

from .__version__ import __version__

# some private imports for create_model function
from typing import Optional as _Optional
import torch as _torch


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: _Optional[str] = "imagenet",
    num_freeze_layers: _Optional[int] = None,
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> _torch.nn.Module:
    """Models entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """

    archs = [
        Unet,
        UnetPlusPlus,
        MAnet,
        Linknet,
        FPN,
        PSPNet,
        DeepLabV3,
        DeepLabV3Plus,
        PAN,
    ]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. Available options are: {}".format(
                arch,
                list(archs_dict.keys()),
            )
        )

    # Create the model
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )

    # Decide which layers to freeze based on the freeze_layers parameter
    if num_freeze_layers is not None:
        if num_freeze_layers < 0 or num_freeze_layers > len(list(model.encoder.children())):
            raise ValueError("Invalid value for freeze_layers.")
        
        # Freeze the specified number of layers
        for i, param in enumerate(model.encoder.parameters()):
            if i < num_freeze_layers:
                param.requires_grad = False
            else:
                break

    return model

