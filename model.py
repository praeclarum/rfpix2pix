from typing import Optional

import torch

from fnn import *

class RFPix2pixModel(nn.Module):
    def __init__(
        self,
        max_size: int,
        sample_batch_size: int,
        train_batch_size: int,
        train_minibatch_size: int,
        train_images: int,
        learning_rate: float,
        num_inference_steps: int,
        timestep_sampling: str,
        velocity_net: Config,
    ):
        super().__init__()
        self.max_size = max_size
        self.sample_batch_size = sample_batch_size
        self.train_batch_size = train_batch_size
        self.train_minibatch_size = train_minibatch_size
        self.train_images = train_images
        self.learning_rate = learning_rate
        self.num_inference_steps = num_inference_steps
        self.timestep_sampling = timestep_sampling
        self.velocity_net = object_from_config(
            velocity_net,
            input_channels=3,
            output_channels=3,
        )

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
            # Standard uniform sampling
            return torch.rand(batch_size, device=device, dtype=dtype)
        elif self.timestep_sampling == "logit-normal":
            # Logit-normal: samples concentrated around t=0.5 (SD3 style)
            # Higher density in the "hard" middle region
            return torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))
        else:
            raise ValueError(f"Unknown timestep_sampling: {self.timestep_sampling}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Training forward pass with flow matching.
        
        Args:
            x: (B, 3, H, W) input image
            t: (B,) timesteps in [0, 1]
            
        Returns:
            v_pred: predicted velocity
        """
        v_pred = self.velocity_net(x, t)
        return v_pred

    @torch.no_grad()
    def generate(
        self,
        x0: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate image from a single input image.
        
        Args:
            x0: (B, 3, H, W) input image
            num_steps: number of flow integration steps (default: self.num_inference_steps)
            
        Returns:
            x1: (B, 3, H, W) generated image
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
            
        B, C, H, W = x0.shape

        z = x0 # Initial state

        # Euler integration with uniform steps
        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            t_val = step_idx * dt
            t = torch.full((B,), t_val, device=x0.device, dtype=x0.dtype)
            
            # Predict velocity
            v = self.velocity_net(z, t)
            
            # Euler step
            z = z + v * dt
        
        return z

register_type("RFPix2pixModel", RFPix2pixModel)
