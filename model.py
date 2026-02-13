from typing import Optional

import torch

from fnn import *
from codec import Codec
from saliency import Saliency

class RFPix2pixModel(nn.Module):
    """
    Top-level RF Pix2Pix model for unpaired image-to-image translation.
    
    Composes three self-contained modules:
        - codec: Encodes/decodes between pixel space and latent space
        - velocity: Predicts flow velocity in latent space
        - saliency: Domain classifier for saliency-weighted loss
    
    Each module owns its own training parameters and can be checkpointed
    independently. This class handles construction-time channel wiring
    and provides inference (generate).
    """
    def __init__(
        self,
        max_size: int,
        codec: Config = {"type": "Codec", "net": {"type": "Identity"}},
        velocity: Config = {"type": "Velocity", "net": {"type": "UNet", "model_channels": 64, "ch_mult": [1, 2, 4]}},
        saliency: Config = {"type": "Saliency", "net": {"type": "ResNetSaliencyNet"}},
        domain0: list = [],
        domain1: list = [],
    ):
        super().__init__()
        self.max_size = max_size
        self.domain0 = domain0
        self.domain1 = domain1
        # Codec: encode/decode between pixel and latent space
        self.codec: Codec = object_from_config(codec, type="Codec")
        latent_ch = self.codec.out_channels
        # Velocity: predicts flow velocity in latent space
        self.velocity: Velocity = object_from_config(
            velocity,
            type="Velocity",
            input_channels=latent_ch,
            output_channels=latent_ch,
        )
        # Saliency: domain classifier for saliency-weighted loss
        self.saliency: Saliency = object_from_config(
            saliency,
            type="Saliency",
            in_channels=latent_ch,
        )
        # Start with codec and saliency network frozen
        self.eval_codec()
        self.eval_saliency()

    def eval_codec(self):
        """Freeze codec for downstream training phases."""
        for param in self.codec.parameters():
            param.requires_grad = False
        self.codec.eval()

    def train_codec(self):
        """Unfreeze codec for codec training phase."""
        for param in self.codec.parameters():
            param.requires_grad = True
        self.codec.train()

    def eval_saliency(self):
        """Freeze saliency network for velocity training phase."""
        for param in self.saliency.parameters():
            param.requires_grad = False
        self.saliency.eval()

    def train_saliency(self):
        """Unfreeze saliency network for saliency training phase."""
        for param in self.saliency.parameters():
            param.requires_grad = True
        self.saliency.train()

    @torch.no_grad()
    def generate(
        self,
        z0: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate image by integrating the velocity ODE and decoding.
        
        Args:
            z0: (B, C, H', W') starting latent
            num_steps: number of flow integration steps (default: velocity.num_inference_steps)
            
        Returns:
            x1: (B, 3, H, W) generated image (pixel space)
        """
        if num_steps is None:
            num_steps = self.velocity.num_inference_steps

        z = z0
        B = z.shape[0]

        # Euler integration in latent space
        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            t_val = step_idx * dt
            t = torch.full((B,), t_val, device=z.device, dtype=z.dtype)
            v = self.velocity(z, t)
            z = z + v * dt
        
        # Decode back to pixel space
        return self.codec.decode(z)

register_type("RFPix2pixModel", RFPix2pixModel)

class Velocity(nn.Module):
    """
    Velocity module holder: wraps a velocity network (UNet/DiT) with training
    and inference parameters.
    
    The net config determines the actual velocity prediction architecture.
    Training parameters control the velocity training phase.
    
    Properties:
        num_downsamples: number of spatial downsampling levels (from net)
    
    Example config:
        {
            "type": "Velocity",
            "net": {"type": "UNet", "model_channels": 96, "ch_mult": [1, 2, 4]},
            "train_batch_size": 48,
            "train_minibatch_size": 24,
            "train_images": 6000000,
            "learning_rate": 0.0002,
            "num_inference_steps": 12,
            "timestep_sampling": "logit-normal",
            "sample_batch_size": 8,
            "structure_pairing": false,
            "structure_candidates": 8
        }
    """
    def __init__(
        self,
        net: Config,
        input_channels: int = 3,
        output_channels: int = 3,
        train_batch_size: int = 48,
        train_minibatch_size: int = 48,
        train_images: int = 6000000,
        learning_rate: float = 0.0002,
        num_inference_steps: int = 12,
        timestep_sampling: str = "logit-normal",
        sample_batch_size: int = 8,
        structure_pairing: bool = False,
        structure_candidates: int = 8,
        bf16: bool = False,
    ):
        super().__init__()
        self.net = object_from_config(
            net,
            input_channels=input_channels,
            output_channels=output_channels,
        )
        self.train_batch_size = train_batch_size
        self.train_minibatch_size = train_minibatch_size
        self.train_images = train_images
        self.learning_rate = learning_rate
        self.num_inference_steps = num_inference_steps
        self.timestep_sampling = timestep_sampling
        self.sample_batch_size = sample_batch_size
        self.structure_pairing = structure_pairing
        self.structure_candidates = structure_candidates
        self.bf16 = bf16

    @property
    def num_downsamples(self) -> int:
        return self.net.num_downsamples

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at interpolated latent state.
        
        Args:
            z_t: (B, C, H, W) interpolated latent
            t: (B,) timesteps in [0, 1]
            
        Returns:
            (B, C, H, W) predicted velocity
        """
        return self.net(z_t, t)

    def sample_timestep(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Sample flow timesteps based on configured strategy.
        
        Args:
            batch_size: number of timesteps to sample
            device: torch device
            dtype: torch dtype
            
        Returns:
            (B,) tensor of timesteps in [0, 1]
        """
        if self.timestep_sampling == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)
        elif self.timestep_sampling == "logit-normal":
            return torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))
        else:
            raise ValueError(f"Unknown timestep_sampling: {self.timestep_sampling}")
register_type("Velocity", Velocity)
