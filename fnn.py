from typing import Any, Callable, Dict
import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# DEVICE
#
# set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`
# to enable fallback to CPU when the MPS device is not available
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Enable TensorFloat32 for better performance on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
print(f"Using device: {device}")


#
# CONFIG
#
Config = Dict[str, Any]
registered_types: Dict[str, Callable] = {}
def register_type(name: str, ctor: Callable):
    registered_types[name] = ctor
def object_from_config(config: Config, **kwargs) -> Any:
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary.")
    t = config["type"] if "type" in config else kwargs.get("type")
    if t in registered_types:
        new_config = dict(config)
        new_config.update(kwargs)
        new_kwargs = dict(new_config)
        del new_kwargs["type"]
        new_object = registered_types[t](**new_kwargs)
        new_object.__config = new_config
        return new_object
    else:
        raise ValueError(f"Unknown config object type: {t}")
def save_config(config: Config, filename: str):
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)
def load_config(filename: str) -> Config:
    with open(filename, "r") as f:
        return json.load(f)
def load_config_object(filename: str) -> Any:
    return object_from_config(load_config(filename))
def save_module(obj: nn.Module, filename: str):
    out_dict = {
        "config": obj.__config,
        "state_dict": obj.state_dict(),
    }
    torch.save(out_dict, filename)


def load_module(filename: str, **kwargs) -> nn.Module:
    # Load to CPU first for cross-device compatibility (CUDA -> MPS, etc.)
    in_dict = torch.load(filename, map_location="cpu", weights_only=False)
    obj = object_from_config(in_dict["config"], **kwargs)
    obj.load_state_dict(in_dict["state_dict"])
    return obj


#
# NN
#
def get_normalization(normalization: str, num_channels: int) -> nn.Module:
    if normalization == "None":
        return nn.Identity()
    if normalization == "BatchNorm1d":
        return nn.BatchNorm1d(num_channels)
    if normalization == "BatchNorm2d":
        return nn.BatchNorm2d(num_channels)
    if normalization == "InstanceNorm1d":
        return nn.InstanceNorm1d(num_channels)
    if normalization == "InstanceNorm2d":
        return nn.InstanceNorm2d(num_channels)
    if normalization == "GroupNorm32":
        return nn.GroupNorm(num_groups=32, num_channels=num_channels)
    if normalization == "GroupNorm16":
        return nn.GroupNorm(num_groups=16, num_channels=num_channels)
    raise NotImplementedError(f"Unknown normalization: {normalization}")

def get_activation(activation: str) -> nn.Module:
    if activation == "None":
        return nn.Identity()
    if activation == "LeakyReLU":
        return nn.LeakyReLU()
    if activation == "ReLU":
        return nn.ReLU()
    if activation == "SiLU":
        return nn.SiLU()
    if activation == "Tanh":
        return nn.Tanh()
    if activation == "Sigmoid":
        return nn.Sigmoid()
    if activation == "Softmax":
        return nn.Softmax()
    raise NotImplementedError(f"Unknown activation: {activation}")

def zero_module(module: nn.Module, should_zero: bool = True):
    """
    Zero out the parameters of a module and return it.
    """
    if should_zero:
        for p in module.parameters():
            p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        normalization: str,
        activation: str,
        input_kernel_size=3,
        kernel_size=3,
        zero_out=False,
    ):
        super().__init__()
        self.in_layers = nn.Sequential(
            get_normalization(normalization, in_channels),
            get_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=input_kernel_size, padding=input_kernel_size//2),
        )
        self.out_layers = nn.Sequential(
            get_normalization(normalization, out_channels),
            get_activation(activation),
            zero_module(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                should_zero=zero_out,
            ),
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        skip = self.skip_connection(x)
        out = skip + h
        return out
register_type("ResBlock", ResBlock)


class SelfAttention2d(nn.Module):
    """
    Spatial self-attention for 2D feature maps.

    Uses only CoreML/ANE-safe operations:
    - 1x1 Conv2d for Q/K/V and output projections
    - reshape + permute for head splitting
    - torch.bmm for attention matmul
    - softmax(dim=-1)

    Applies pre-norm (GroupNorm) and a zero-initialized output projection
    for stable residual learning: output = x + proj(attn(norm(x))).
    """

    def __init__(self, channels: int, num_heads: int = 8, normalization: str = "GroupNorm32"):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5  # Pre-compute for FP16 safety

        self.norm = get_normalization(normalization, channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature map

        Returns:
            (B, C, H, W) output with attended features added as residual
        """
        B, C, H, W = x.shape
        N = H * W  # number of spatial tokens

        h = self.norm(x)

        # Project to Q, K, V using 1x1 convolutions: (B, C, H, W)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Reshape to (B*heads, N, head_dim) for batched matmul
        # From (B, C, H, W) -> (B, heads, head_dim, N) -> (B*heads, N, head_dim)
        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).reshape(B * self.num_heads, N, self.head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).reshape(B * self.num_heads, N, self.head_dim)

        # Scale Q before matmul for FP16 numerical stability
        q = q * self.scale

        # Attention: (B*heads, N, N)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values: (B*heads, N, head_dim)
        out = torch.bmm(attn, v)

        # Reshape back to (B, C, H, W)
        # (B*heads, N, head_dim) -> (B, heads, N, head_dim) -> (B, heads, head_dim, N) -> (B, C, H, W)
        out = out.reshape(B, self.num_heads, N, self.head_dim).permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Zero-initialized output projection + residual
        out = self.out_proj(out)
        return x + out


#
# MODEL
#
class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        model_channels: int,
        ch_mult: list[int],
        normalization: str,
        activation: str,
        num_res_blocks: int,
        zero_res_blocks: bool = False,
        attention_resolutions: list[int] = [],
        num_attention_heads: int = 8,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.model_channels = model_channels
        self.num_downsamples = len(ch_mult)
        self.attention_resolutions = attention_resolutions

        # Timestep embedding MLP
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_block = nn.Sequential(
            nn.Conv2d(input_channels, model_channels, kernel_size=5, padding=2),
        )
        enc_blocks = []
        dec_blocks = []
        down_blocks = []
        up_blocks = []
        time_proj_enc = []  # project time embedding to each encoder level
        time_proj_dec = []  # project time embedding to each decoder level
        ch = model_channels
        time_embed_dim = model_channels * 4
        def res_block(in_ch, out_ch):
            blocks = [ResBlock(in_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks))
            return nn.Sequential(*blocks)
        enc_attn = []
        dec_attn = []
        for i, ch_m in enumerate(ch_mult):
            out_ch = model_channels * ch_m
            enc_blocks.append(res_block(ch, out_ch))
            time_proj_enc.append(nn.Linear(time_embed_dim, out_ch))
            skip_ch = out_ch
            prev_ch = 0 if i == len(ch_mult) - 1 else model_channels * ch_mult[i + 1]
            if i < len(ch_mult) - 1:
                down_blocks.append(nn.AvgPool2d(2))
            dec_blocks.append(res_block(skip_ch + prev_ch, out_ch))
            time_proj_dec.append(nn.Linear(time_embed_dim, out_ch))
            should_upsample = i > 0
            up_blocks.append(nn.UpsamplingBilinear2d(scale_factor=2) if should_upsample else nn.Identity())
            # Self-attention at specified resolution levels
            if i in attention_resolutions:
                enc_attn.append(SelfAttention2d(out_ch, num_heads=num_attention_heads, normalization=normalization))
                dec_attn.append(SelfAttention2d(out_ch, num_heads=num_attention_heads, normalization=normalization))
            else:
                enc_attn.append(nn.Identity())
                dec_attn.append(nn.Identity())
            ch = out_ch
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.time_proj_enc = nn.ModuleList(time_proj_enc)
        self.time_proj_dec = nn.ModuleList(time_proj_dec)
        self.enc_attn = nn.ModuleList(enc_attn)
        self.dec_attn = nn.ModuleList(dec_attn)
        self.output_block = nn.Sequential(
            res_block(model_channels, model_channels),
            get_normalization(normalization, model_channels),
            get_activation(activation),
            nn.Conv2d(model_channels, output_channels, kernel_size=3, padding=1),
        )
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            t: (B,) flow timestep in [0, 1]
            
        Returns:
            (B, C_out, H, W) predicted velocity
        """
        # Timestep embedding
        t_emb = get_timestep_embedding(t, self.model_channels)  # (B, model_channels)
        t_emb = self.time_embed(t_emb)  # (B, time_embed_dim)

        h = self.input_block(x)
        res_hs = []

        # Encoder
        for i, enc_block in enumerate(self.enc_blocks):
            h = enc_block(h)
            # Add time embedding (broadcast over H, W)
            t_proj = self.time_proj_enc[i](t_emb)[:, :, None, None]  # (B, C, 1, 1)
            h = h + t_proj
            h = self.enc_attn[i](h)
            res_hs.append(h)
            if i < len(self.down_blocks):
                h = self.down_blocks[i](h)

        # Decoder
        i = len(self.dec_blocks) - 1
        while i >= 0:
            if i < len(self.dec_blocks) - 1:
                # Upsample FIRST, then concatenate with skip connection
                h = self.up_blocks[i + 1](h)
                h = self.dec_blocks[i](torch.cat([h, res_hs[i]], dim=1))
            else:
                # Bottleneck: no skip connection, no upsampling before
                h = self.dec_blocks[i](h)
            # Add time embedding
            t_proj = self.time_proj_dec[i](t_emb)[:, :, None, None]
            h = h + t_proj
            h = self.dec_attn[i](h)
            i -= 1

        # Final upsample (from first encoder level, if needed)
        h = self.up_blocks[0](h)
        y = self.output_block(h)
        return y
register_type("UNet", UNet)

#
# LOSS
#
class L1Loss(nn.Module):
    def __init__(self, id: str="l1_loss", weight: float=1.0):
        super().__init__()
        self.id = id
        self.weight = weight
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(input, target)
register_type("L1Loss", L1Loss)


class MSELoss(nn.Module):
    def __init__(self, id: str="mse_loss", weight: float=1.0):
        super().__init__()
        self.id = id
        self.weight = weight
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target)
register_type("MSELoss", MSELoss)


#
# CODEC (Identity for now, VAE later)
#
class Codec(nn.Module):
    """Base class for encoder/decoder pairs. Subclass for VAE."""
    def __init__(self, latent_channels: int, spatial_factor: int = 1):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_factor = spatial_factor  # 1 = no downscale, 8 = 8x downscale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, H, W) -> (B, C, H', W')"""
        raise NotImplementedError()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, C, H', W') -> (B, 3, H, W)"""
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (for reconstruction loss)."""
        return self.decode(self.encode(x))


class IdentityCodec(Codec):
    """Pass-through codec for pixel-space training. No compression."""
    def __init__(self):
        super().__init__(latent_channels=3, spatial_factor=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z
register_type("IdentityCodec", IdentityCodec)


#
# 3D BLOCKS (for temporal models)
#
class ResBlock3D(nn.Module):
    """3D ResBlock with Conv3d. Mirrors ResBlock but with temporal dimension."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str,
        activation: str,
        kernel_size: int = 3,
        zero_out: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        # GroupNorm works on channels, agnostic to spatial dims
        self.in_layers = nn.Sequential(
            get_normalization(normalization, in_channels),
            get_activation(activation),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )
        self.out_layers = nn.Sequential(
            get_normalization(normalization, out_channels),
            get_activation(activation),
            zero_module(
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                should_zero=zero_out,
            ),
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W)"""
        h = self.in_layers(x)
        h = self.out_layers(h)
        skip = self.skip_connection(x)
        return skip + h
register_type("ResBlock3D", ResBlock3D)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings (same as DDPM / Transformer positional encoding).
    
    Args:
        timesteps: (B,) tensor of timestep values in [0, 1]
        embedding_dim: dimension of the output embedding
        
    Returns:
        (B, embedding_dim) tensor of embeddings
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps[:, None].float() * emb[None, :] * 1000.0  # scale to match diffusion convention
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class UNet3D(nn.Module):
    """
    3D UNet for temporal sequence generation.
    Downsamples spatially only, preserves temporal dimension.
    Supports timestep conditioning via additive embedding.
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        model_channels: int,
        ch_mult: list[int],
        normalization: str,
        activation: str,
        num_res_blocks: int,
        zero_res_blocks: bool = False,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.model_channels = model_channels
        self.num_downsamples = len(ch_mult)

        # Timestep embedding MLP
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        self.input_block = nn.Conv3d(input_channels, model_channels, kernel_size=3, padding=1)

        enc_blocks = []
        dec_blocks = []
        down_blocks = []
        up_blocks = []
        time_proj_enc = []  # project time embedding to each encoder level
        time_proj_dec = []  # project time embedding to each decoder level

        ch = model_channels

        def res_block(in_ch: int, out_ch: int):
            blocks = [ResBlock3D(in_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock3D(out_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks))
            return nn.Sequential(*blocks)

        for i, ch_m in enumerate(ch_mult):
            out_ch = model_channels * ch_m
            enc_blocks.append(res_block(ch, out_ch))
            time_proj_enc.append(nn.Linear(time_embed_dim, out_ch))
            
            skip_ch = out_ch
            prev_ch = 0 if i == len(ch_mult) - 1 else model_channels * ch_mult[i + 1]
            
            if i < len(ch_mult) - 1:
                # Downsample spatially only (kernel 1,2,2 stride 1,2,2)
                down_blocks.append(nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            
            dec_blocks.append(res_block(skip_ch + prev_ch, out_ch))
            time_proj_dec.append(nn.Linear(time_embed_dim, out_ch))
            
            should_upsample = i > 0
            if should_upsample:
                # Upsample spatially only
                up_blocks.append(nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'))
            else:
                up_blocks.append(nn.Identity())
            
            ch = out_ch

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.time_proj_enc = nn.ModuleList(time_proj_enc)
        self.time_proj_dec = nn.ModuleList(time_proj_dec)

        self.output_block = nn.Sequential(
            res_block(model_channels, model_channels),
            get_normalization(normalization, model_channels),
            get_activation(activation),
            nn.Conv3d(model_channels, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input tensor (noisy sequence + conditioning)
            t: (B,) flow timestep in [0, 1]
            
        Returns:
            (B, C_out, T, H, W) predicted velocity
        """
        # Timestep embedding
        t_emb = get_timestep_embedding(t, self.model_channels)  # (B, model_channels)
        t_emb = self.time_embed(t_emb)  # (B, time_embed_dim)

        h = self.input_block(x)
        res_hs = []
        
        # Encoder
        for i, enc_block in enumerate(self.enc_blocks):
            h = enc_block(h)
            # Add time embedding (broadcast over T, H, W)
            t_proj = self.time_proj_enc[i](t_emb)[:, :, None, None, None]  # (B, C, 1, 1, 1)
            h = h + t_proj
            res_hs.append(h)
            if i < len(self.down_blocks):
                h = self.down_blocks[i](h)

        # Decoder
        i = len(self.dec_blocks) - 1
        while i >= 0:
            if i < len(self.dec_blocks) - 1:
                # Upsample FIRST, then concatenate with skip connection
                h = self.up_blocks[i + 1](h)
                h = self.dec_blocks[i](torch.cat([h, res_hs[i]], dim=1))
            else:
                # Bottleneck: no skip connection, no upsampling before
                h = self.dec_blocks[i](h)
            # Add time embedding
            t_proj = self.time_proj_dec[i](t_emb)[:, :, None, None, None]
            h = h + t_proj
            i -= 1
        
        # Final upsample (from first encoder level, if needed)
        h = self.up_blocks[0](h)

        y = self.output_block(h)
        return y
register_type("UNet3D", UNet3D)
