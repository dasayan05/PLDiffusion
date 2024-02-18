from typing import Tuple, Optional, Union

from diffusers.models import unet_2d


class UNet2DModel(unet_2d.UNet2DModel):

    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):

        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            time_embedding_type,
            freq_shift,
            flip_sin_to_cos,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            mid_block_scale_factor,
            downsample_padding,
            downsample_type,
            upsample_type,
            dropout,
            act_fn,
            attention_head_dim,
            norm_num_groups,
            attn_norm_num_groups,
            norm_eps,
            resnet_time_scale_shift,
            add_attention,
            class_embed_type,
            num_class_embeds,
            num_train_timesteps
        )
