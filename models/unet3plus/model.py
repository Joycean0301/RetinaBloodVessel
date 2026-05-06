import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Union

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from segmentation_models_pytorch.encoders import get_encoder

from .decoder import Unet3PlusDecoder


class Unet3Plus(SegmentationModel):
    """
    UNet 3+ for semantic segmentation.

    UNet 3+ uses full-scale skip connections. Each decoder stage fuses feature
    maps from all encoder scales and deeper decoder stages, which helps combine
    fine vessel details with high-level context.
    """

    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (320, 320, 320, 320, 320),
        decoder_cat_channels: int = 64,
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_depth != 5:
            raise ValueError("Unet3Plus currently supports encoder_depth=5.")

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "The usage of decoder_use_batchnorm is deprecated. Please modify your code for decoder_use_norm",
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = Unet3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            cat_channels=decoder_cat_channels,
            use_norm=decoder_use_norm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=2,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unet3plus-{}".format(encoder_name)
        self.initialize()
