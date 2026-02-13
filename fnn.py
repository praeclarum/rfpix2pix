from typing import Any, Callable, Dict, Optional
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
# UNet
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
        output_activation: str = "None",
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, H, W) -> (B, latent_channels, H/sf, W/sf)"""
        raise NotImplementedError()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, latent_channels, H', W') -> (B, 3, H, W)"""
        raise NotImplementedError()

    def encode_params(self, x: torch.Tensor) -> dict:
        """
        Encode and return all distribution parameters.
        
        For deterministic nets, returns {"z": z}.
        For VAE nets, returns {"z": z, "mu": mu, "logvar": logvar}.
        """
        return {"z": self.encode(x)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def freeze_backbone(self):
        """Override in nets with pretrained backbones."""
        pass

    def unfreeze_backbone(self):
        """Override in nets with pretrained backbones."""
        pass

    def has_backbone(self) -> bool:
        """Return True if this net has a pretrained backbone that supports warmup."""
        return False

    def get_optimizer_param_groups(self, lr: float, backbone_frozen: bool) -> list:
        """Get parameter groups for optimizer. Override for differential LR."""
        return [{"params": self.parameters(), "lr": lr}]


class Identity(CodecNet):
    """Pass-through: no encoding/decoding. latent_channels=3, spatial_factor=1."""
    def __init__(self):
        super().__init__(latent_channels=3, spatial_factor=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

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
                layers.append(SelfAttention2d(out_ch, num_heads=num_attention_heads, normalization=normalization))
            if i < len(ch_mult) - 1:
                layers.append(get_downsample(downsample, out_ch))
            ch = out_ch
        # Mid block
        layers.append(ResBlock(ch, ch, normalization=normalization, activation=activation))
        if mid_attention:
            layers.append(SelfAttention2d(ch, num_heads=num_attention_heads, normalization=normalization))
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
            layers.append(SelfAttention2d(ch, num_heads=num_attention_heads, normalization=normalization))
            layers.append(ResBlock(ch, ch, normalization=normalization, activation=activation))
        # Upsample levels (reversed)
        for i in reversed(range(len(ch_mult))):
            out_ch = model_channels * ch_mult[i]
            layers.append(ResBlock(ch, out_ch, normalization=normalization, activation=activation))
            for _ in range(num_res_blocks - 1):
                layers.append(ResBlock(out_ch, out_ch, normalization=normalization, activation=activation))
            if i in attention_resolutions:
                layers.append(SelfAttention2d(out_ch, num_heads=num_attention_heads, normalization=normalization))
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
        """Number of channels the encoder should produce. Overridden by VAE."""
        return latent_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_params(x)["z"]
register_type("VAE", VAE)


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


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity using VGG16 features.
    
    Computes L1 distance between feature maps at multiple layers.
    Input is expected in [-1, 1] range (converted to [0, 1] internally).
    The VGG backbone is always frozen.
    """
    kind = "reconstruction"
    def __init__(self, id: str = "lpips_loss", weight: float = 1.0):
        super().__init__()
        self.id = id
        self.weight = weight
        self._vgg: Optional[nn.Module] = None
        self._feature_layers: list[str] = []
        # Lazily initialized to avoid loading VGG at config parse time

    def _ensure_vgg(self, device: torch.device):
        if self._vgg is not None:
            return
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self._vgg = vgg
        # Feature extraction points (after ReLU):
        # relu1_2 = layer 3, relu2_2 = layer 8, relu3_3 = layer 15, relu4_3 = layer 22
        self._feature_indices = [3, 8, 15, 22]

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale VGG features from images in [0, 1]."""
        features = []
        h = x
        for i, layer in enumerate(self._vgg):  # type: ignore
            h = layer(h)
            if i in self._feature_indices:
                features.append(h)
            if i >= self._feature_indices[-1]:
                break
        return features

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            input: (B, 3, H, W) reconstructed images in [-1, 1]
            target: (B, 3, H, W) original images in [-1, 1]
        """
        self._ensure_vgg(input.device)
        # Convert [-1, 1] -> [0, 1]
        inp_01 = (input + 1.0) * 0.5
        tgt_01 = (target + 1.0) * 0.5
        # Extract features and compute L1 distance per layer
        feats_inp = self._extract_features(inp_01)
        feats_tgt = self._extract_features(tgt_01)
        loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)
        for fi, ft in zip(feats_inp, feats_tgt):
            loss = loss + F.l1_loss(fi, ft)
        return loss

    def state_dict(self, *args, **kwargs):
        """
        Exclude VGG weights from state_dict.
        
        VGG is frozen and lazily loaded from torchvision,
        so we don't need to save its weights.
        """
        full_state = super().state_dict(*args, **kwargs)
        # Filter out _vgg weights
        filtered_state = {
            k: v for k, v in full_state.items()
            if not k.startswith("_vgg.")
        }
        return filtered_state

register_type("LPIPSLoss", LPIPSLoss)
