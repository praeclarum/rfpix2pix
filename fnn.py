from typing import Any, Callable, Dict, Optional, Sequence, cast
import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


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
def load_module(filename: str, strict: bool = True, **kwargs) -> nn.Module:
    # Load to CPU first for cross-device compatibility (CUDA -> MPS, etc.)
    in_dict = torch.load(filename, map_location="cpu", weights_only=False)
    obj = object_from_config(in_dict["config"], **kwargs)
    obj.load_state_dict(in_dict["state_dict"], strict=strict)
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

def get_downsample(downsample: str, channels: int) -> nn.Module:
    if downsample == "AvgPool":
        return nn.AvgPool2d(2)
    if downsample == "StridedConv":
        return nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    raise NotImplementedError(f"Unknown downsample: {downsample}")

def get_upsample(upsample: str, channels: int) -> nn.Module:
    if upsample == "Bilinear":
        return nn.UpsamplingBilinear2d(scale_factor=2)
    if upsample == "NearestConv":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    raise NotImplementedError(f"Unknown upsample: {upsample}")

def get_downsample_3d(downsample: str, channels: int) -> nn.Module:
    if downsample == "AvgPool":
        return nn.AvgPool3d(2)
    if downsample == "StridedConv":
        return nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
    raise NotImplementedError(f"Unknown downsample: {downsample}")

def get_upsample_3d(upsample: str, channels: int) -> nn.Module:
    if upsample == "Bilinear":
        return nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
    if upsample == "NearestConv":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
    raise NotImplementedError(f"Unknown upsample: {upsample}")

def get_downsample_td(downsample: str, channels: int) -> nn.Module:
    """Temporal-preserving downsample for video tensors (B, C, T, H, W)."""
    if downsample == "AvgPool":
        return nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    if downsample == "StridedConv":
        return nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    raise NotImplementedError(f"Unknown downsample: {downsample}")

def get_upsample_td(upsample: str, channels: int) -> nn.Module:
    """Temporal-preserving upsample for video tensors (B, C, T, H, W)."""
    if upsample == "Bilinear":
        return nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
    if upsample == "NearestConv":
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )
    raise NotImplementedError(f"Unknown upsample: {upsample}")

def zero_module(module: nn.Module, should_zero: bool = True):
    """
    Zero out the parameters of a module and return it.
    """
    if should_zero:
        for p in module.parameters():
            p.detach().zero_()
    return module

class TimestepBlock(nn.Module):
    """Marker base class for modules that accept `(x, t_emb)` in forward."""

    def forward(self, x, t_emb):
        raise NotImplementedError()

class TimestepSequential(nn.Sequential):
    """
    Sequential container that forwards timestep embeddings only to layers
    that support it (instances of `TimestepBlock`).
    """

    def forward(self, input, t_emb=None):
        x = input
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x

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


#
# 2D BLOCKS (image models)
#
class ResBlock(TimestepBlock):
    """Residual block with optional FiLM time conditioning.

    - If `time_embed_dim > 0`, `forward(x, t_emb)` applies FiLM modulation.
    - If `time_embed_dim == 0`, the block is unconditional and ignores `t_emb`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization: str,
        activation: str,
        input_kernel_size=3,
        kernel_size=3,
        zero_out=False,
        time_embed_dim: int = 0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.in_layers = nn.Sequential(
            get_normalization(normalization, in_channels),
            get_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=input_kernel_size, padding=input_kernel_size//2),
        )
        self.out_norm = get_normalization(normalization, out_channels)
        self.out_act = get_activation(activation)
        self.out_conv = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            should_zero=zero_out,
        )
        if time_embed_dim > 0:
            # Project time embedding to per-channel scale and shift
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * out_channels),
            )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb=None):
        h = self.in_layers(x)
        h = self.out_norm(h)
        if self.time_embed_dim > 0 and t_emb is not None:
            # FiLM: modulate the norm output with learned scale + shift
            scale_shift = self.time_proj(t_emb)  # (B, 2*C)
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.out_act(h)
        h = self.out_conv(h)
        return self.skip_connection(x) + h
register_type("ResBlock", ResBlock)


class SelfAttention(nn.Module):
    """
    Spatial self-attention for 2D feature maps.

    Applies pre-norm (GroupNorm), multi-head scaled dot-product attention,
    and a zero-initialized output projection for stable residual learning:
    output = x + proj(attn(norm(x))).
    """

    def __init__(self, channels: int, num_heads: int = 8, normalization: str = "GroupNorm32"):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = get_normalization(normalization, channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = x

        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        n = h * w
        q = q.view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.out_proj(out)
        return r + out


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
        output_activation: str = "None",
        downsample: str = "AvgPool",
        upsample: str = "Bilinear",
        mid_attention: bool = False,
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
        ch = model_channels
        time_embed_dim = model_channels * 4
        film_dim = time_embed_dim
        def res_block(in_ch, out_ch):
            blocks = [ResBlock(in_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim))
            return TimestepSequential(*blocks)
        def res_block_plain(in_ch, out_ch):
            """Unconditional ResBlock stack (no time conditioning)."""
            blocks = [ResBlock(in_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks))
            return nn.Sequential(*blocks)
        enc_attn = []
        dec_attn = []
        for i, ch_m in enumerate(ch_mult):
            out_ch = model_channels * ch_m
            enc_blocks.append(res_block(ch, out_ch))
            skip_ch = out_ch
            prev_ch = 0 if i == len(ch_mult) - 1 else model_channels * ch_mult[i + 1]
            if i < len(ch_mult) - 1:
                down_blocks.append(get_downsample(downsample, out_ch))
            dec_blocks.append(res_block(skip_ch + prev_ch, out_ch))
            should_upsample = i > 0
            up_blocks.append(get_upsample(upsample, out_ch) if should_upsample else nn.Identity())
            # Self-attention at specified resolution levels
            if i in attention_resolutions:
                enc_attn.append(SelfAttention(out_ch, num_heads=num_attention_heads, normalization=normalization))
                dec_attn.append(SelfAttention(out_ch, num_heads=num_attention_heads, normalization=normalization))
            else:
                enc_attn.append(nn.Identity())
                dec_attn.append(nn.Identity())
            ch = out_ch
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.enc_attn = nn.ModuleList(enc_attn)
        self.dec_attn = nn.ModuleList(dec_attn)
        # Mid block: ResBlock -> SelfAttention -> ResBlock (SD-style bottleneck)
        self.has_mid_block = mid_attention
        if mid_attention:
            deepest_ch = model_channels * ch_mult[-1]
            self.mid_res1 = ResBlock(deepest_ch, deepest_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim)
            self.mid_attn = SelfAttention(deepest_ch, num_heads=num_attention_heads, normalization=normalization)
            self.mid_res2 = ResBlock(deepest_ch, deepest_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim)
        self.output_block = nn.Sequential(
            res_block_plain(model_channels, model_channels),
            get_normalization(normalization, model_channels),
            get_activation(activation),
            nn.Conv2d(model_channels, output_channels, kernel_size=3, padding=1),
            get_activation(output_activation),
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
            h = enc_block(h, t_emb)
            h = self.enc_attn[i](h)
            res_hs.append(h)
            if i < len(self.down_blocks):
                h = self.down_blocks[i](h)

        # Mid block
        if self.has_mid_block:
            h = self.mid_res1(h, t_emb)
            h = self.mid_attn(h)
            h = self.mid_res2(h, t_emb)

        # Decoder
        i = len(self.dec_blocks) - 1
        while i >= 0:
            if i < len(self.dec_blocks) - 1:
                # Upsample FIRST, then concatenate with skip connection
                h = self.up_blocks[i + 1](h)
                h_in = torch.cat([h, res_hs[i]], dim=1)
            else:
                # Bottleneck: no skip connection, no upsampling before
                h_in = h
            h = self.dec_blocks[i](h_in, t_emb)
            h = self.dec_attn[i](h)
            i -= 1

        # Final upsample (from first encoder level, if needed)
        h = self.up_blocks[0](h)
        y = self.output_block(h)
        return y
register_type("UNet", UNet)


#
# DiT (Diffusion Transformer)
#
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift and scale to normalized token sequences.
    x: (B, T, D), shift/scale: (B, D) → broadcast over T."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    """
    Compute 2D sin-cos positional embeddings on the fly.

    Args:
        embed_dim: total embedding dimension (must be divisible by 2; each spatial
                   axis uses embed_dim//2).
        grid_h: number of patch rows
        grid_w: number of patch columns
        device: target device

    Returns:
        (grid_h * grid_w, embed_dim) tensor of positional embeddings
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half = embed_dim // 2
    # Build 1-D frequency bands (half // 2 freqs per axis)
    omega = torch.arange(half // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (half // 2)))
    # Grid positions
    pos_h = torch.arange(grid_h, device=device, dtype=torch.float32)
    pos_w = torch.arange(grid_w, device=device, dtype=torch.float32)
    # Outer products → (grid, freqs)
    out_h = pos_h[:, None] * omega[None, :]  # (grid_h, half//2)
    out_w = pos_w[:, None] * omega[None, :]  # (grid_w, half//2)
    # Sin/cos interleave for each axis → (grid, half)
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (grid_h, half)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (grid_w, half)
    # Broadcast to 2-D grid: (grid_h, grid_w, embed_dim)
    emb = torch.cat([
        emb_h[:, None, :].expand(-1, grid_w, -1),
        emb_w[None, :, :].expand(grid_h, -1, -1),
    ], dim=2)  # (grid_h, grid_w, embed_dim)
    return emb.reshape(grid_h * grid_w, embed_dim)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Uses pre-norm LayerNorm (no learned affine), with per-sample adaptive
    shift, scale, and gate regressed from the conditioning vector.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Pre-norms (un-affine)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention (QKV fused projection)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=True)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # adaLN modulation: produces 6 vectors (shift, scale, gate) x2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) token sequence
            c: (B, D) conditioning vector
        Returns:
            (B, T, D) output tokens
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # --- Attention path ---
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        B, T, D = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv.unbind(0)
        h = F.scaled_dot_product_attention(q, k, v)  # (B, heads, T, head_dim)
        h = h.transpose(1, 2).reshape(B, T, D)
        h = self.attn_out(h)
        x = x + gate_msa.unsqueeze(1) * h

        # --- MLP path ---
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiTFinalLayer(nn.Module):
    """
    Final layer of DiT: adaLN modulation (shift + scale only, no gate)
    followed by a linear projection to patch_size^2 * out_channels.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for velocity prediction.

    Drop-in replacement for UNet:  forward(x, t) -> velocity, same shapes.
    Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022).

    Config parameters:
        patch_size   (int)  : patch tokenisation stride (image dims must be divisible)
        hidden_size  (int)  : transformer width
        depth        (int)  : number of DiTBlocks
        num_heads    (int)  : attention heads (must divide hidden_size)
        mlp_ratio    (float): MLP hidden dim = hidden_size * mlp_ratio
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_size: int = 2,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        # Reuse the existing divisibility infrastructure:
        # data.py computes image_size_multiple = 2 ** num_downsamples
        self.num_downsamples = int(math.log2(patch_size)) if patch_size > 1 else 0

        # --- Patch embedding (Conv2d acts as linear projection of flattened patches) ---
        self.x_embedder = nn.Conv2d(
            input_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

        # --- Timestep conditioning ---
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # --- Final layer ---
        self.final_layer = DiTFinalLayer(hidden_size, patch_size, output_channels)

        # Cached positional embeddings (lazily populated in forward)
        self._cached_pos_emb: Optional[torch.Tensor] = None
        self._cached_pos_hw: tuple[int, int] = (0, 0)

        self.initialize_weights()

    def initialize_weights(self):
        """Weight init following the DiT paper."""
        # Xavier uniform for all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Patch embedding: treat Conv2d as linear
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0) # pyright: ignore[reportArgumentType]

        # Timestep MLP
        nn.init.normal_(self.t_embedder[0].weight, std=0.02) # pyright: ignore[reportArgumentType]
        nn.init.normal_(self.t_embedder[2].weight, std=0.02) # pyright: ignore[reportArgumentType]

        # Zero-out adaLN modulation outputs in every block
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0) # type: ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0) # type: ignore

        # Zero-out final layer outputs
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0) # type: ignore
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0) # type: ignore
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Reshape patch tokens back to a spatial image.

        Args:
            x: (B, T, patch_size**2 * C_out)
            h: number of patch rows
            w: number of patch columns
        Returns:
            (B, C_out, H, W) image tensor
        """
        p = self.patch_size
        c = self.output_channels
        # (B, h, w, p, p, c) -> (B, c, h, p, w, p) -> (B, c, h*p, w*p)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            t: (B,) flow timestep in [0, 1]

        Returns:
            (B, C_out, H, W) predicted velocity
        """
        p = self.patch_size

        # Patchify: (B, C, H, W) -> (B, D, H', W') -> (B, T, D)
        x = self.x_embedder(x)                        # (B, D, H', W')
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)               # (B, T, D)

        # Add 2-D sin-cos positional embedding (cached for repeated resolutions)
        if self._cached_pos_hw != (Hp, Wp) or self._cached_pos_emb is None or self._cached_pos_emb.device != x.device:
            self._cached_pos_emb = get_2d_sincos_pos_embed(self.hidden_size, Hp, Wp, device=x.device)
            self._cached_pos_hw = (Hp, Wp)
        x = x + self._cached_pos_emb.unsqueeze(0)       # broadcast over batch

        # Timestep conditioning
        c = get_timestep_embedding(t, self.hidden_size) # (B, D)
        c = self.t_embedder(c)                          # (B, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)                             # (B, T, D)

        # Final projection + unpatchify
        x = self.final_layer(x, c)                      # (B, T, p*p*C_out)
        x = self.unpatchify(x, Hp, Wp)                  # (B, C_out, H, W)
        return x
register_type("DiT", DiT)


#
# CODEC NETS
#
class CodecNet(nn.Module):
    """
    Abstract base for codec encoder/decoder networks.
    
    Subclasses must implement encode(), decode(), and encode_params().
    The Codec holder class delegates to these methods.
    """
    def __init__(self, latent_channels: int, spatial_factor: int):
        super().__init__()
        self._latent_channels = latent_channels
        self._spatial_factor = spatial_factor

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def spatial_factor(self) -> int:
        return self._spatial_factor

    def sample_random_latent(self, shape: Sequence[int], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, H, W) -> (B, latent_channels, H/sf, W/sf)"""
        return self.encode_params(x)["z"]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, latent_channels, H', W') -> (B, 3, H, W)"""
        raise NotImplementedError()

    def encode_params(self, x: torch.Tensor) -> dict:
        """
        Encode and return all distribution parameters.
        
        For deterministic nets, returns {"z": z}.
        For VAE nets, returns {"z": z, "mu": mu, "logvar": logvar}.
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class Identity(CodecNet):
    """Pass-through: no encoding/decoding. latent_channels=3, spatial_factor=1."""
    def __init__(self):
        super().__init__(latent_channels=3, spatial_factor=1)

    def encode_params(self, x: torch.Tensor) -> dict:
        return {"z": x}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z
register_type("Identity", Identity)


class Encoder(nn.Module):
    """
    Convolutional encoder: downsamples spatially and projects to a latent space.

    Architecture: Conv -> [ResBlock x N -> Attn? -> Downsample] x levels -> mid ResBlock(s) + Attn? -> norm -> act -> Conv

    spatial_factor = 2 ** (len(ch_mult) - 1)

    Downsample modes:
        "AvgPool"     — AvgPool2d(2) (default)
        "StridedConv" — Conv2d 3×3 stride 2
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        model_channels: int = 64,
        ch_mult: list[int] = [1, 2, 4],
        normalization: str = "GroupNorm32",
        activation: str = "SiLU",
        num_res_blocks: int = 2,
        attention_resolutions: list[int] = [],
        mid_attention: bool = False,
        num_attention_heads: int = 8,
        downsample: str = "AvgPool",
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)]
        ch = model_channels
        for i, m in enumerate(ch_mult):
            out_ch = model_channels * m
            layers.append(ResBlock(ch, out_ch, normalization=normalization, activation=activation))
            for _ in range(num_res_blocks - 1):
                layers.append(ResBlock(out_ch, out_ch, normalization=normalization, activation=activation))
            if i in attention_resolutions:
                layers.append(SelfAttention(out_ch, num_heads=num_attention_heads, normalization=normalization))
            if i < len(ch_mult) - 1:
                layers.append(get_downsample(downsample, out_ch))
            ch = out_ch
        # Mid block
        layers.append(ResBlock(ch, ch, normalization=normalization, activation=activation))
        if mid_attention:
            layers.append(SelfAttention(ch, num_heads=num_attention_heads, normalization=normalization))
            layers.append(ResBlock(ch, ch, normalization=normalization, activation=activation))
        # Project to latent
        layers.append(get_normalization(normalization, ch))
        layers.append(get_activation(activation))
        layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
register_type("Encoder", Encoder)


class Decoder(nn.Module):
    """
    Convolutional decoder: upsamples spatially from a latent space.

    Architecture: Conv -> mid ResBlock(s) + Attn? -> [ResBlock x N -> Attn? -> Upsample] x levels -> norm -> act -> Conv -> output_activation?

    spatial_factor = 2 ** (len(ch_mult) - 1)

    Upsample modes:
        "Bilinear"      — UpsamplingBilinear2d(2x) (default)
        "NearestConv"   — Nearest-neighbor 2x + Conv2d 3x3
    """
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        model_channels: int = 64,
        ch_mult: list[int] = [1, 2, 4],
        normalization: str = "GroupNorm32",
        activation: str = "SiLU",
        num_res_blocks: int = 2,
        attention_resolutions: list[int] = [],
        mid_attention: bool = False,
        num_attention_heads: int = 8,
        upsample: str = "Bilinear",
        output_activation: str = "Tanh",
    ):
        super().__init__()
        ch = model_channels * ch_mult[-1]
        layers: list[nn.Module] = [nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)]
        # Mid block
        layers.append(ResBlock(ch, ch, normalization=normalization, activation=activation))
        if mid_attention:
            layers.append(SelfAttention(ch, num_heads=num_attention_heads, normalization=normalization))
            layers.append(ResBlock(ch, ch, normalization=normalization, activation=activation))
        # Upsample levels (reversed)
        for i in reversed(range(len(ch_mult))):
            out_ch = model_channels * ch_mult[i]
            layers.append(ResBlock(ch, out_ch, normalization=normalization, activation=activation))
            for _ in range(num_res_blocks - 1):
                layers.append(ResBlock(out_ch, out_ch, normalization=normalization, activation=activation))
            if i in attention_resolutions:
                layers.append(SelfAttention(out_ch, num_heads=num_attention_heads, normalization=normalization))
            ch = out_ch
            if i > 0:
                layers.append(get_upsample(upsample, ch))
        # To pixel space
        layers.append(get_normalization(normalization, model_channels))
        layers.append(get_activation(activation))
        layers.append(nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1))
        layers.append(get_activation(output_activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)
register_type("Decoder", Decoder)


class AutoEncoder(CodecNet):
    """
    Deterministic convolutional autoencoder.

    Composes an Encoder and Decoder without any stochastic sampling.
    spatial_factor = 2 ** (len(ch_mult) - 1)
    """
    def __init__(
        self,
        latent_channels: int = 16,
        model_channels: int = 64,
        ch_mult: list[int] = [1, 2, 4],
        normalization: str = "GroupNorm32",
        activation: str = "SiLU",
        num_res_blocks: int = 2,
        decoder_num_res_blocks: Optional[int] = None,
        attention_resolutions: list[int] = [],
        mid_attention: bool = False,
        num_attention_heads: int = 8,
        downsample: str = "AvgPool",
        upsample: str = "Bilinear",
        output_activation: str = "Tanh",
    ):
        spatial_factor = 2 ** (len(ch_mult) - 1)
        super().__init__(latent_channels=latent_channels, spatial_factor=spatial_factor)
        self.encoder = Encoder(
            in_channels=3,
            out_channels=self._encoder_out_channels(latent_channels),
            model_channels=model_channels,
            ch_mult=ch_mult,
            normalization=normalization,
            activation=activation,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            mid_attention=mid_attention,
            num_attention_heads=num_attention_heads,
            downsample=downsample,
        )
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=3,
            model_channels=model_channels,
            ch_mult=ch_mult,
            normalization=normalization,
            activation=activation,
            num_res_blocks=decoder_num_res_blocks if decoder_num_res_blocks is not None else num_res_blocks,
            attention_resolutions=attention_resolutions,
            mid_attention=mid_attention,
            num_attention_heads=num_attention_heads,
            upsample=upsample,
            output_activation=output_activation,
        )

    def _encoder_out_channels(self, latent_channels: int) -> int:
        """Number of channels the encoder should produce."""
        return latent_channels

    def encode_params(self, x: torch.Tensor) -> dict:
        return {"z": self.encoder(x)}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
register_type("AutoEncoder", AutoEncoder)


class VAE(AutoEncoder):
    """
    Variational autoencoder.

    Inherits from AutoEncoder but doubles the encoder output channels to
    produce (mu, logvar) and applies the reparameterization trick.
    """

    def _encoder_out_channels(self, latent_channels: int) -> int:
        return 2 * latent_channels

    def encode_params(self, x: torch.Tensor) -> dict:
        """Encode and return mu, logvar, and reparameterized sample z."""
        h = self.encoder(x)  # (B, 2*latent_channels, H', W')
        mu, logvar = h.chunk(2, dim=1)  # each (B, latent_channels, H', W')
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return {"z": z, "mu": mu, "logvar": logvar}
register_type("VAE", VAE)


class SAE(AutoEncoder):
    """
    Spherical autoencoder.

    Maps the latent space to a sphere.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        noise_angle_degrees_max: float = 85.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.noise_angle_degrees_max = noise_angle_degrees_max
        self.sigma_max = math.tan(math.radians(noise_angle_degrees_max))

    def spherify(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent vectors onto a sphere."""
        return z * torch.rsqrt(z.square().mean(dim=[1, 2, 3], keepdim=True) + self.eps)

    def sample_random_latent(self, shape: Sequence[int], device: torch.device) -> torch.Tensor:
        z = torch.randn(shape, device=device)
        return self.spherify(z)

    def encode_params(self, x: torch.Tensor) -> dict:
        """Encode and return mu, logvar, and reparameterized sample z."""
        z: torch.Tensor = self.encoder(x)  # (B, latent_channels, H', W')
        v = self.spherify(z)  # project to sphere
        if self.training:
            b, _, _, _ = v.shape
            sigma = torch.rand(b, 1, 1, 1, device=z.device) * self.sigma_max
            sigma_sub = torch.rand(b, 1, 1, 1, device=z.device) * 0.5 * sigma
            e = torch.randn_like(v)
            v_noisy_big = self.spherify(v + sigma * e)
            v_noisy_small = self.spherify(v + sigma_sub * e)
            return {"z": v_noisy_small, "v_noisy_big": v_noisy_big, "v_clean": v}
        else:
            return {"z": v, "v_noisy_big": v, "v_clean": v}
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        v = self.spherify(z)
        return super().decode(v)
register_type("SAE", SAE)


#
# 3D BLOCKS (volumetric models)
#
class ResBlock3d(TimestepBlock):
    """3D residual block with optional FiLM time conditioning.

    - If `time_embed_dim > 0`, `forward(x, t_emb)` applies FiLM modulation.
    - If `time_embed_dim == 0`, the block is unconditional and ignores `t_emb`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization: str,
        activation: str,
        input_kernel_size=3,
        kernel_size=3,
        zero_out=False,
        time_embed_dim: int = 0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.in_layers = nn.Sequential(
            get_normalization(normalization, in_channels),
            get_activation(activation),
            nn.Conv3d(in_channels, out_channels, kernel_size=input_kernel_size, padding=input_kernel_size//2),
        )
        self.out_norm = get_normalization(normalization, out_channels)
        self.out_act = get_activation(activation)
        self.out_conv = zero_module(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            should_zero=zero_out,
        )
        if time_embed_dim > 0:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * out_channels),
            )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_norm(h)
        if self.time_embed_dim > 0 and t_emb is not None:
            scale_shift = self.time_proj(t_emb)  # (B, 2*C)
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.out_act(h)
        h = self.out_conv(h)
        return self.skip_connection(x) + h
register_type("ResBlock3d", ResBlock3d)


class SelfAttention3d(nn.Module):
    """
    Self-attention for volumetric 3D feature maps.

    Applies pre-norm (GroupNorm), multi-head scaled dot-product attention,
    and a zero-initialized output projection for stable residual learning:
    output = x + proj(attn(norm(x))).
    """

    def __init__(self, channels: int, num_heads: int = 8, normalization: str = "GroupNorm32"):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = get_normalization(normalization, channels)
        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.out_proj = zero_module(nn.Conv3d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        r = x

        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        n = d * h * w
        q = q.view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.permute(0, 1, 3, 2).reshape(b, c, d, h, w)
        out = self.out_proj(out)
        return r + out


class UNet3d(nn.Module):
    """
    Volumetric 3D UNet.

    Mirrors UNet with Conv3d/attention3d blocks and FiLM timestep conditioning.
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
        attention_resolutions: list[int] = [],
        num_attention_heads: int = 8,
        output_activation: str = "None",
        downsample: str = "AvgPool",
        upsample: str = "Bilinear",
        mid_attention: bool = False,
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
            nn.Conv3d(input_channels, model_channels, kernel_size=5, padding=2),
        )

        enc_blocks = []
        dec_blocks = []
        down_blocks = []
        up_blocks = []
        enc_attn = []
        dec_attn = []
        ch = model_channels
        film_dim = time_embed_dim

        def res_block(in_ch: int, out_ch: int):
            blocks = [ResBlock3d(in_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock3d(out_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim))
            return TimestepSequential(*blocks)

        def res_block_plain(in_ch: int, out_ch: int):
            blocks = [ResBlock3d(in_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock3d(out_ch, out_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks))
            return nn.Sequential(*blocks)

        for i, ch_m in enumerate(ch_mult):
            out_ch = model_channels * ch_m
            enc_blocks.append(res_block(ch, out_ch))

            skip_ch = out_ch
            prev_ch = 0 if i == len(ch_mult) - 1 else model_channels * ch_mult[i + 1]

            if i < len(ch_mult) - 1:
                down_blocks.append(get_downsample_3d(downsample, out_ch))

            dec_blocks.append(res_block(skip_ch + prev_ch, out_ch))

            should_upsample = i > 0
            up_blocks.append(get_upsample_3d(upsample, out_ch) if should_upsample else nn.Identity())

            if i in attention_resolutions:
                enc_attn.append(SelfAttention3d(out_ch, num_heads=num_attention_heads, normalization=normalization))
                dec_attn.append(SelfAttention3d(out_ch, num_heads=num_attention_heads, normalization=normalization))
            else:
                enc_attn.append(nn.Identity())
                dec_attn.append(nn.Identity())

            ch = out_ch

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.enc_attn = nn.ModuleList(enc_attn)
        self.dec_attn = nn.ModuleList(dec_attn)

        self.has_mid_block = mid_attention
        if mid_attention:
            deepest_ch = model_channels * ch_mult[-1]
            self.mid_res1 = ResBlock3d(deepest_ch, deepest_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim)
            self.mid_attn = SelfAttention3d(deepest_ch, num_heads=num_attention_heads, normalization=normalization)
            self.mid_res2 = ResBlock3d(deepest_ch, deepest_ch, normalization=normalization, activation=activation, zero_out=zero_res_blocks, time_embed_dim=film_dim)

        self.output_block = nn.Sequential(
            res_block_plain(model_channels, model_channels),
            get_normalization(normalization, model_channels),
            get_activation(activation),
            nn.Conv3d(model_channels, output_channels, kernel_size=3, padding=1),
            get_activation(output_activation),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) input tensor
            t: (B,) flow timestep in [0, 1]
            
        Returns:
            (B, C_out, D, H, W) predicted velocity
        """
        # Timestep embedding
        t_emb = get_timestep_embedding(t, self.model_channels)  # (B, model_channels)
        t_emb = self.time_embed(t_emb)  # (B, time_embed_dim)

        h = self.input_block(x)
        res_hs = []

        # Encoder
        for i, enc_block in enumerate(self.enc_blocks):
            h = enc_block(h, t_emb)
            h = self.enc_attn[i](h)
            res_hs.append(h)
            if i < len(self.down_blocks):
                h = self.down_blocks[i](h)

        if self.has_mid_block:
            h = self.mid_res1(h, t_emb)
            h = self.mid_attn(h)
            h = self.mid_res2(h, t_emb)

        # Decoder
        i = len(self.dec_blocks) - 1
        while i >= 0:
            if i < len(self.dec_blocks) - 1:
                # Upsample FIRST, then concatenate with skip connection
                h = self.up_blocks[i + 1](h)
                h_in = torch.cat([h, res_hs[i]], dim=1)
            else:
                # Bottleneck: no skip connection, no upsampling before
                h_in = h
            h = self.dec_blocks[i](h_in, t_emb)
            h = self.dec_attn[i](h)
            i -= 1

        # Final upsample (from first encoder level, if needed)
        h = self.up_blocks[0](h)

        y = self.output_block(h)
        return y
register_type("UNet3d", UNet3d)


#
# TD BLOCKS (temporal models)
#
class TemporalAttentionTd(nn.Module):
    """Temporal self-attention for video tensors with per-pixel tokenization over T."""

    def __init__(self, channels: int, num_heads: int = 8, normalization: str = "GroupNorm32"):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = get_normalization(normalization, channels)
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.out_proj = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        r = x
        x = self.norm(x)

        # (B, C, T, H, W) -> (B*H*W, C, T)
        seq = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)

        q = self.q_proj(seq)
        k = self.k_proj(seq)
        v = self.v_proj(seq)

        q = q.view(b * h * w, self.num_heads, self.head_dim, t).permute(0, 1, 3, 2)
        k = k.view(b * h * w, self.num_heads, self.head_dim, t).permute(0, 1, 3, 2)
        v = v.view(b * h * w, self.num_heads, self.head_dim, t).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.permute(0, 1, 3, 2).reshape(b * h * w, c, t)
        out = self.out_proj(out)
        out = out.reshape(b, h, w, c, t).permute(0, 3, 4, 1, 2)
        return r + out


class SpatialAttentionTd(nn.Module):
    """Spatial self-attention for video tensors, applied frame-wise."""

    def __init__(self, channels: int, num_heads: int = 8, normalization: str = "GroupNorm32"):
        super().__init__()
        self.attn2d = SelfAttention(channels, num_heads=num_heads, normalization=normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.attn2d(y)
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class SelfAttentionTd(nn.Module):
    """Video attention block with optional spatial and temporal sub-attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        normalization: str = "GroupNorm32",
        spatial_attention: bool = True,
        temporal_attention: bool = True,
    ):
        super().__init__()
        self.spatial_attn = SpatialAttentionTd(channels, num_heads=num_heads, normalization=normalization) if spatial_attention else nn.Identity()
        self.temporal_attn = TemporalAttentionTd(channels, num_heads=num_heads, normalization=normalization) if temporal_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        return x


class ResBlockTd(TimestepBlock):
    """Temporal-aware residual block for video tensors.

    Compared to `ResBlock3d`, this block treats the 3rd axis as time and uses
    factorized convolutions to keep temporal/spatial mixing distinct:
    - Spatial convs use kernels `(1, k, k)`.
    - Temporal convs use kernels `(k_t, 1, 1)`.
    - Down/up sampling in `UNetTd` preserves temporal length.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization: str,
        activation: str,
        input_kernel_size=3,
        kernel_size=3,
        temporal_kernel_size: int = 3,
        zero_out=False,
        time_embed_dim: int = 0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.in_layers = nn.Sequential(
            get_normalization(normalization, in_channels),
            get_activation(activation),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, input_kernel_size, input_kernel_size),
                padding=(0, input_kernel_size // 2, input_kernel_size // 2),
            ),
        )
        self.in_temporal = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0),
        )

        self.out_norm = get_normalization(normalization, out_channels)
        self.out_act = get_activation(activation)
        self.out_spatial = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2),
        )
        self.out_temporal = zero_module(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1, 1),
                padding=(temporal_kernel_size // 2, 0, 0),
            ),
            should_zero=zero_out,
        )

        if time_embed_dim > 0:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * out_channels),
            )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.in_temporal(h)
        h = self.out_norm(h)
        if self.time_embed_dim > 0 and t_emb is not None:
            scale_shift = self.time_proj(t_emb)  # (B, 2*C)
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.out_act(h)
        h = self.out_spatial(h)
        h = self.out_temporal(h)
        return self.skip_connection(x) + h
register_type("ResBlockTd", ResBlockTd)


class UNetTd(nn.Module):
    """Temporal-aware UNet for video generation.

    Compared to `UNet3d` (volumetric):
    - Interprets the 3rd axis as time `(B, C, T, H, W)`.
    - Down/upsamples spatial axes only, preserving sequence length.
    - Uses `ResBlockTd` and `SelfAttentionTd` (spatial + temporal attention).
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
        attention_resolutions: list[int] = [],
        num_attention_heads: int = 8,
        output_activation: str = "None",
        downsample: str = "AvgPool",
        upsample: str = "Bilinear",
        mid_attention: bool = False,
        temporal_kernel_size: int = 3,
        spatial_attention: bool = True,
        temporal_attention: bool = True,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.model_channels = model_channels
        self.num_downsamples = len(ch_mult)
        self.attention_resolutions = attention_resolutions

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_block = nn.Sequential(
            nn.Conv3d(input_channels, model_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
        )

        enc_blocks = []
        dec_blocks = []
        down_blocks = []
        up_blocks = []
        enc_attn = []
        dec_attn = []
        ch = model_channels
        film_dim = time_embed_dim

        def res_block(in_ch, out_ch):
            blocks = [ResBlockTd(
                in_ch,
                out_ch,
                normalization=normalization,
                activation=activation,
                zero_out=zero_res_blocks,
                time_embed_dim=film_dim,
                temporal_kernel_size=temporal_kernel_size,
            )]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlockTd(
                    out_ch,
                    out_ch,
                    normalization=normalization,
                    activation=activation,
                    zero_out=zero_res_blocks,
                    time_embed_dim=film_dim,
                    temporal_kernel_size=temporal_kernel_size,
                ))
            return TimestepSequential(*blocks)

        def res_block_plain(in_ch, out_ch):
            blocks = [ResBlockTd(
                in_ch,
                out_ch,
                normalization=normalization,
                activation=activation,
                zero_out=zero_res_blocks,
                temporal_kernel_size=temporal_kernel_size,
            )]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlockTd(
                    out_ch,
                    out_ch,
                    normalization=normalization,
                    activation=activation,
                    zero_out=zero_res_blocks,
                    temporal_kernel_size=temporal_kernel_size,
                ))
            return nn.Sequential(*blocks)

        for i, ch_m in enumerate(ch_mult):
            out_ch = model_channels * ch_m
            enc_blocks.append(res_block(ch, out_ch))
            skip_ch = out_ch
            prev_ch = 0 if i == len(ch_mult) - 1 else model_channels * ch_mult[i + 1]
            if i < len(ch_mult) - 1:
                down_blocks.append(get_downsample_td(downsample, out_ch))
            dec_blocks.append(res_block(skip_ch + prev_ch, out_ch))
            should_upsample = i > 0
            up_blocks.append(get_upsample_td(upsample, out_ch) if should_upsample else nn.Identity())
            if i in attention_resolutions:
                enc_attn.append(SelfAttentionTd(
                    out_ch,
                    num_heads=num_attention_heads,
                    normalization=normalization,
                    spatial_attention=spatial_attention,
                    temporal_attention=temporal_attention,
                ))
                dec_attn.append(SelfAttentionTd(
                    out_ch,
                    num_heads=num_attention_heads,
                    normalization=normalization,
                    spatial_attention=spatial_attention,
                    temporal_attention=temporal_attention,
                ))
            else:
                enc_attn.append(nn.Identity())
                dec_attn.append(nn.Identity())
            ch = out_ch

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.enc_attn = nn.ModuleList(enc_attn)
        self.dec_attn = nn.ModuleList(dec_attn)

        self.has_mid_block = mid_attention
        if mid_attention:
            deepest_ch = model_channels * ch_mult[-1]
            self.mid_res1 = ResBlockTd(
                deepest_ch,
                deepest_ch,
                normalization=normalization,
                activation=activation,
                zero_out=zero_res_blocks,
                time_embed_dim=film_dim,
                temporal_kernel_size=temporal_kernel_size,
            )
            self.mid_attn = SelfAttentionTd(
                deepest_ch,
                num_heads=num_attention_heads,
                normalization=normalization,
                spatial_attention=spatial_attention,
                temporal_attention=temporal_attention,
            )
            self.mid_res2 = ResBlockTd(
                deepest_ch,
                deepest_ch,
                normalization=normalization,
                activation=activation,
                zero_out=zero_res_blocks,
                time_embed_dim=film_dim,
                temporal_kernel_size=temporal_kernel_size,
            )

        self.output_block = nn.Sequential(
            res_block_plain(model_channels, model_channels),
            get_normalization(normalization, model_channels),
            get_activation(activation),
            nn.Conv3d(model_channels, output_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            get_activation(output_activation),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input tensor
            t: (B,) flow timestep in [0, 1]

        Returns:
            (B, C_out, T, H, W) predicted velocity
        """
        t_emb = get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        h = self.input_block(x)
        res_hs = []

        for i, enc_block in enumerate(self.enc_blocks):
            h = enc_block(h, t_emb)
            h = self.enc_attn[i](h)
            res_hs.append(h)
            if i < len(self.down_blocks):
                h = self.down_blocks[i](h)

        if self.has_mid_block:
            h = self.mid_res1(h, t_emb)
            h = self.mid_attn(h)
            h = self.mid_res2(h, t_emb)

        i = len(self.dec_blocks) - 1
        while i >= 0:
            if i < len(self.dec_blocks) - 1:
                h = self.up_blocks[i + 1](h)
                h_in = torch.cat([h, res_hs[i]], dim=1)
            else:
                h_in = h
            h = self.dec_blocks[i](h_in, t_emb)
            h = self.dec_attn[i](h)
            i -= 1

        h = self.up_blocks[0](h)
        y = self.output_block(h)
        return y
register_type("UNetTd", UNetTd)


#
# LOSS
#
class L1Loss(nn.Module):
    kind = "reconstruction"
    def __init__(self, id: str="l1_loss", weight: float=1.0):
        super().__init__()
        self.id = id
        self.weight = weight
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(input, target)
register_type("L1Loss", L1Loss)


class MSELoss(nn.Module):
    kind = "reconstruction"
    def __init__(self, id: str="mse_loss", weight: float=1.0):
        super().__init__()
        self.id = id
        self.weight = weight
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target)
register_type("MSELoss", MSELoss)


class KLLoss(nn.Module):
    """
    KL divergence loss for VAE: -0.5 * mean(1 + logvar - mu^2 - exp(logvar)).
    
    This is a regularization loss — it takes (mu, logvar) not (input, target).
    """
    kind = "regularization"
    def __init__(self, id: str = "kl_loss", weight: float = 1e-6):
        super().__init__()
        self.id = id
        self.weight = weight
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
register_type("KLLoss", KLLoss)


_LPIPS_BACKBONE_CACHE: Dict[tuple[str, str], tuple[nn.Sequential, tuple[int, ...]]] = {}

def _get_lpips_backbone(backbone: str, device: torch.device) -> tuple[nn.Sequential, tuple[int, ...]]:
    """Get (or lazily create) a frozen LPIPS feature backbone for (backbone, device)."""
    key = (backbone, str(device))
    model_and_feature_indices = _LPIPS_BACKBONE_CACHE.get(key)
    if model_and_feature_indices is None:
        if backbone == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            model = cast(nn.Sequential, vgg16(weights=VGG16_Weights.DEFAULT).features.to(device))
            feature_indices = (3, 8, 15, 22)  # conv1_2, conv2_2, conv3_3, conv4_3 feature map indices
        else:
            raise ValueError(f"Unsupported LPIPS backbone: {backbone}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model_and_feature_indices = (model, feature_indices)
        _LPIPS_BACKBONE_CACHE[key] = model_and_feature_indices
    return model_and_feature_indices

def _extract_lpips_features(
    x: torch.Tensor,
    model_and_feature_indices: tuple[nn.Sequential, tuple[int, ...]],
) -> list[torch.Tensor]:
    """Extract LPIPS feature maps from a configured backbone."""
    model, feature_indices = model_and_feature_indices
    features = []
    h = x
    last_idx = feature_indices[-1]
    for i, layer in enumerate(model.children()):
        h = layer(h)
        if i in feature_indices:
            features.append(h)
        if i >= last_idx:
            break
    return features

def lpips_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    backbone: str = "vgg16",
) -> torch.Tensor:
    """
    Functional LPIPS-style perceptual loss.

    Args:
        input: (B, 3, H, W) reconstructed images in [-1, 1]
        target: (B, 3, H, W) target images in [-1, 1]
        backbone: Feature backbone name (currently supports "vgg16")
    """
    inp_01 = (input + 1.0) * 0.5
    tgt_01 = (target + 1.0) * 0.5
    model_and_feature_indices = _get_lpips_backbone(backbone, input.device)
    feats_inp = _extract_lpips_features(inp_01, model_and_feature_indices)
    feats_tgt = _extract_lpips_features(tgt_01, model_and_feature_indices)
    loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)
    for fi, ft in zip(feats_inp, feats_tgt):
        loss = loss + F.l1_loss(fi, ft)
    return loss

class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity using VGG16 features.
    
    Computes L1 distance between feature maps at multiple layers.
    Input is expected in [-1, 1] range (converted to [0, 1] internally).
    The VGG backbone is always frozen.
    """
    kind = "reconstruction"
    def __init__(
        self,
        id: str = "lpips_loss",
        weight: float = 1.0,
        backbone: str = "vgg16",
    ):
        super().__init__()
        self.id = id
        self.weight = weight
        self.backbone = backbone

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return lpips_loss(
            input,
            target,
            backbone=self.backbone,
        )

register_type("LPIPSLoss", LPIPSLoss)


class SAEPixelConsistencyLoss(nn.Module):
    kind = "sae"
    def __init__(
        self,
        id: str = "pix_con_loss",
        weight: float = 1.0,
        backbone: str = "vgg16",
        recon_weight: float = 1.0,
        lpips_weight: float = 1.0,
    ):
        super().__init__()
        self.id = id
        self.weight = weight
        self.backbone = backbone
        self.recon_weight = recon_weight
        self.lpips_weight = lpips_weight

    def forward(self, x_hat_noisy_small: torch.Tensor, x_hat_noisy_big: torch.Tensor, **kwargs) -> torch.Tensor:
        # target is sg(xhat_noisy_small) where sg is torch.stop_gradient, but since LPIPS backbone is frozen
        target = x_hat_noisy_small.detach()
        lpips = lpips_loss(
            x_hat_noisy_big,
            target,
            backbone=self.backbone,
        )
        recon = F.l1_loss(x_hat_noisy_big, target)
        return lpips * self.lpips_weight + recon * self.recon_weight

register_type("SAEPixelConsistencyLoss", SAEPixelConsistencyLoss)


class SAELatentConsistencyLoss(nn.Module):
    kind = "sae"
    def __init__(
        self,
        id: str = "lat_con_loss",
        weight: float = 1.0,
    ):
        super().__init__()
        self.id = id
        self.weight = weight

    def cosine_similarity_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_flat = a.view(a.size(0), -1)
        b_flat = b.view(b.size(0), -1)
        a_norm = F.normalize(a_flat, dim=1)
        b_norm = F.normalize(b_flat, dim=1)
        cos_sim = (a_norm * b_norm).sum(dim=1)
        loss = 1.0 - cos_sim
        return loss.mean()

    def forward(self, v_x_hat_noisy_big: torch.Tensor, v_clean: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.cosine_similarity_loss(v_x_hat_noisy_big, v_clean)

register_type("SAELatentConsistencyLoss", SAELatentConsistencyLoss)


#
# TRAINING UTILITIES
#

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, schedule: str) -> float:
    """Compute learning rate for a given step with warmup + schedule."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return max_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if schedule == "cosine":
        # Cosine decay from max_lr -> 0 over remaining steps
        return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported LR schedule: {schedule}")


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of weights on CPU to save GPU memory.
    Only moves to GPU when swapping in for sampling.
    """
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cpu()
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * param."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                # Move shadow to device, update in-place, move back to CPU
                self.shadow[name].lerp_(param.data.cpu(), 1.0 - self.decay)
    
    @contextmanager
    def swap(self, model: torch.nn.Module):
        """
        Context manager: temporarily swap EMA weights into the model for inference.
        Restores training weights on exit.
        """
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
    
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        for name in self.shadow:
            if name in state_dict:
                self.shadow[name] = state_dict[name].cpu()

