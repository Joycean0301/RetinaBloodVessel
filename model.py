import torch.nn as nn

from models import Unet, UnetPlusPlus, Unet3Plus


# ────────────────────────── MODEL FACTORY ───────────────────────
def build_model(model_name: str, backbone: str = 'vgg16') -> nn.Module:
    """Build a segmentation model from the local models package."""
    kwargs = dict(
        encoder_name=backbone,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,   # raw logits; loss handles sigmoid
        encoder_depth=5,
        decoder_attention_type=None,
    )

    if model_name == 'Unet':
        return Unet(**kwargs)
    elif model_name == 'UnetPlusPlus':
        return UnetPlusPlus(**kwargs)
    elif model_name == 'Unet3Plus':
        return Unet3Plus(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
