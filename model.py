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
        saliency_net: Config,
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
        self.saliency_net = object_from_config(saliency_net)
        self.saliency_channels = self.saliency_net.output_channels
        # Start with saliency network frozen (alternating training phases)
        self.eval_saliency()

    def eval_saliency(self):
        """Freeze saliency network for velocity training phase."""
        for param in self.saliency_net.parameters():
            param.requires_grad = False
        self.saliency_net.eval()

    def train_saliency(self):
        """Unfreeze saliency network for saliency training phase."""
        for param in self.saliency_net.parameters():
            param.requires_grad = True
        self.saliency_net.train()

    def get_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get domain classification logits from saliency network.
        
        Args:
            x: (B, 3, H, W) input images
            
        Returns:
            (B, num_classes) classification logits (typically 2 for domain 0/1)
        """
        return self.saliency_net(x)

    def compute_saliency_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> dict:
        """
        Compute classification loss for training the saliency network.
        
        The saliency network learns to distinguish domain 0 from domain 1.
        This provides meaningful gradients for the style-transfer loss.
        
        Args:
            x0: (B, 3, H, W) images from domain 0
            x1: (B, 3, H, W) images from domain 1
            
        Returns:
            dict with:
                - loss: scalar cross-entropy loss
                - logits_0: (B, num_classes) logits for domain 0
                - logits_1: (B, num_classes) logits for domain 1
                - accuracy: classification accuracy
        """
        B = x0.shape[0]
        
        # Get classification logits for both domains
        logits_0 = self.get_saliency(x0)  # (B, num_classes)
        logits_1 = self.get_saliency(x1)  # (B, num_classes)
        
        # Concatenate logits and create labels
        # Domain 0 -> label 0, Domain 1 -> label 1
        logits = torch.cat([logits_0, logits_1], dim=0)  # (2B, num_classes)
        labels = torch.cat([
            torch.zeros(B, dtype=torch.long, device=x0.device),
            torch.ones(B, dtype=torch.long, device=x1.device),
        ], dim=0)  # (2B,)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
        
        return {
            "loss": loss,
            "logits_0": logits_0,
            "logits_1": logits_1,
            "accuracy": accuracy,
        }

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

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Training forward pass with flow matching.
        
        Computes the interpolated state x_t and predicts the velocity field.
        For rectified flow: x_t = t * x1 + (1 - t) * x0, target velocity = x1 - x0
        
        Args:
            x0: (B, 3, H, W) source domain image
            x1: (B, 3, H, W) target domain image
            t: (B,) timesteps in [0, 1]
            
        Returns:
            dict with:
                - v_pred: (B, 3, H, W) predicted velocity
                - v_target: (B, 3, H, W) target velocity (x1 - x0)
                - x_t: (B, 3, H, W) interpolated state
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_broadcast = t[:, None, None, None]
        
        # Linear interpolation: x_t = t * x1 + (1 - t) * x0
        x_t = t_broadcast * x1 + (1 - t_broadcast) * x0
        
        # Target velocity for rectified flow
        v_target = x1 - x0
        
        # Predict velocity at interpolated state
        v_pred = self.velocity_net(x_t, t)
        
        return {
            "v_pred": v_pred,
            "v_target": v_target,
            "x_t": x_t,
        }

    def compute_loss(self, x0: torch.Tensor, x1: torch.Tensor, t: Optional[torch.Tensor] = None) -> dict:
        """
        Compute saliency-weighted rectified flow matching loss.
        
        From the paper: min_v ∫ E[||∇h(x_t)^T * (x1 - x0 - v(x_t, t))||²] dt
        
        The saliency network h(x) provides a feature mapping, and ∇h(x_t)^T acts
        as a saliency score that re-weights coordinates so the loss focuses on
        penalizing errors that cause significant changes in feature space.
        
        Args:
            x0: (B, 3, H, W) source domain image
            x1: (B, 3, H, W) target domain image  
            t: (B,) timesteps in [0, 1], if None will be sampled
            
        Returns:
            dict with:
                - loss: scalar saliency-weighted MSE loss
                - v_pred: predicted velocity
                - v_target: target velocity
                - jvp_result: (B, saliency_channels) JVP result for analysis
        """
        from torch.autograd.functional import jvp
        
        B = x0.shape[0]
        
        if t is None:
            t = self.sample_timestep(B, x0.device, x0.dtype)
        
        out = self.forward(x0, x1, t)
        v_pred = out["v_pred"]
        v_target = out["v_target"]
        x_t = out["x_t"]
        
        # Velocity error: v_target - v_pred
        # NOTE: v_pred has gradients connected to velocity_net
        v_error = v_target - v_pred  # (B, 3, H, W)
        
        # Compute saliency-weighted loss: ||J_h(x_t) @ v_error||²
        # where J_h is the Jacobian of saliency_net
        #
        # The JVP computes: J_h @ v_error, giving shape (B, saliency_channels)
        # 
        # GRADIENT FLOW:
        # - x_t is detached: saliency_net params won't receive gradients from x_t
        # - v_error retains grad_fn: backprop flows through v_error -> v_pred -> velocity_net
        # - create_graph=True: ensures the JVP op is in the computation graph
        
        def saliency_latent_fn(x):
            # Use get_latent to get the feature representation h(x)
            # saliency_net should be frozen (requires_grad=False on params)
            return self.saliency_net.get_latent(x)
        
        # Compute JVP: J_h(x_t) @ v_error
        # x_t.detach() ensures no gradient flows through the "primal" input
        # v_error is the "tangent" and retains its gradient connection
        _, jvp_result = jvp(
            saliency_latent_fn,
            (x_t.detach(),),  # primal: detached interpolated state
            (v_error,),       # tangent: velocity error (has gradients!)
            create_graph=True # keep in computation graph for backprop
        )
        # jvp_result: (B, saliency_channels) - velocity error in feature space
        
        # Loss: MSE of saliency-weighted velocity error
        loss = (jvp_result ** 2).mean() # pyright: ignore[reportOperatorIssue]
        
        return {
            "loss": loss,
            "v_pred": v_pred,
            "v_target": v_target,
            "jvp_result": jvp_result,
        }

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
