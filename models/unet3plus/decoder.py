from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class CenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        super().__init__(conv1, conv2)


class FullScaleFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        cat_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        self.interpolation_mode = interpolation_mode
        self.projections = nn.ModuleList(
            [
                md.Conv2dReLU(
                    ch,
                    cat_channels,
                    kernel_size=3,
                    padding=1,
                    use_norm=use_norm,
                )
                for ch in in_channels
            ]
        )
        self.attention = md.Attention(
            attention_type,
            in_channels=cat_channels * len(in_channels),
        )
        self.fuse = md.Conv2dReLU(
            cat_channels * len(in_channels),
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[2:] == size:
            return x
        return F.interpolate(x, size=size, mode=self.interpolation_mode)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        resized_features = [
            self._resize(project(feature), target_size)
            for project, feature in zip(self.projections, features)
        ]
        x = torch.cat(resized_features, dim=1)
        x = self.attention(x)
        return self.fuse(x)


class Unet3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        cat_channels: int = 64,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
        center: bool = False,
    ):
        super().__init__()

        if n_blocks != 5:
            raise ValueError("Unet3PlusDecoder currently supports encoder_depth=5.")

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        # Remove first skip with same spatial resolution, same convention as Unet/Unet++.
        encoder_channels = list(encoder_channels[1:])
        e1_ch, e2_ch, e3_ch, e4_ch, e5_ch = encoder_channels
        d5_ch, d4_ch, d3_ch, d2_ch, d1_ch = decoder_channels

        if center:
            self.center = CenterBlock(e5_ch, d5_ch, use_norm=use_norm)
        else:
            self.center = md.Conv2dReLU(
                e5_ch,
                d5_ch,
                kernel_size=3,
                padding=1,
                use_norm=use_norm,
            )

        kwargs = dict(
            cat_channels=cat_channels,
            use_norm=use_norm,
            attention_type=attention_type,
            interpolation_mode=interpolation_mode,
        )

        self.d4 = FullScaleFusionBlock(
            [e1_ch, e2_ch, e3_ch, e4_ch, d5_ch],
            out_channels=d4_ch,
            **kwargs,
        )
        self.d3 = FullScaleFusionBlock(
            [e1_ch, e2_ch, e3_ch, d4_ch, d5_ch],
            out_channels=d3_ch,
            **kwargs,
        )
        self.d2 = FullScaleFusionBlock(
            [e1_ch, e2_ch, d3_ch, d4_ch, d5_ch],
            out_channels=d2_ch,
            **kwargs,
        )
        self.d1 = FullScaleFusionBlock(
            [e1_ch, d2_ch, d3_ch, d4_ch, d5_ch],
            out_channels=d1_ch,
            **kwargs,
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[1:]
        e1, e2, e3, e4, e5 = features

        d5 = self.center(e5)
        d4 = self.d4([e1, e2, e3, e4, d5], target_size=e4.shape[2:])
        d3 = self.d3([e1, e2, e3, d4, d5], target_size=e3.shape[2:])
        d2 = self.d2([e1, e2, d3, d4, d5], target_size=e2.shape[2:])
        d1 = self.d1([e1, d2, d3, d4, d5], target_size=e1.shape[2:])

        return d1
