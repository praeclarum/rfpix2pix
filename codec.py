import torch
import torch.nn as nn
import torch.nn.functional as F

from fnn import *
from data import ImageAugmentation

class Codec(nn.Module):
    """
    General codec holder: wraps a CodecNet and a list of losses.
    
    The net config determines the actual encoder/decoder architecture.
    Losses define how the codec is trained (empty = no training needed).
    Training parameters (learning_rate, train_images, warmup_fraction, gradient_clip)
    control the codec training phase.
    
    Properties:
        out_channels: number of latent channels (for downstream nets)
        spatial_factor: spatial downsampling factor
    
    Example configs:
        Identity (no-op):
            {"net": {"type": "Identity"}}
        
        VAE:
            {
                "net": {"type": "VAE", "latent_channels": 16, "ch_mult": [1,2,4,8]},
                "losses": [
                    {"type": "L1Loss", "weight": 1.0},
                    {"type": "LPIPSLoss", "weight": 1.0},
                    {"type": "KLLoss", "weight": 1e-6}
                ],
                "learning_rate": 1e-4,
                "train_images": 2000000
            }
    """
    def __init__(
        self,
        net: Config = {"type": "Identity"},
        losses: list[Config] = [],
        learning_rate: float = 1e-4,
        train_images: int = 0,
        train_batch_size: int = 64,
        train_minibatch_size: Optional[int] = None,
        sample_batch_size: int = 8,
        augmentations: list[str] = [],
        gradient_clip: float = 1.0,
        warmup_fraction: float = 0.0,
        lr_schedule: str = "cosine",
    ):
        super().__init__()
        self.net: CodecNet = object_from_config(net)
        self.loss_fns: nn.ModuleList = nn.ModuleList([object_from_config(l) for l in losses])
        self.learning_rate = learning_rate
        self.train_images = train_images
        self.train_batch_size = train_batch_size
        self.train_minibatch_size = train_minibatch_size if train_minibatch_size is not None else train_batch_size
        self.sample_batch_size = sample_batch_size
        self.augmentations = augmentations
        self.augment = ImageAugmentation(augmentations)
        self.gradient_clip = gradient_clip
        self.warmup_fraction = warmup_fraction
        self.lr_schedule = lr_schedule

    @property
    def out_channels(self) -> int:
        """Number of latent channels — used by velocity_net and saliency_net."""
        return self.net.latent_channels

    @property
    def spatial_factor(self) -> int:
        """Spatial downsampling factor."""
        return self.net.spatial_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        return self.net.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space."""
        return self.net.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (reconstruction)."""
        return self.net(x)

    def compute_loss(self, x: torch.Tensor) -> dict:
        """
        Compute all codec losses on input images.
        
        Returns dict with:
            - "loss": weighted sum of all losses
            - Each loss's id: individual unweighted loss value
        """
        params = self.net.encode_params(x)
        z = params["z"]
        x_hat = self.net.decode(z)

        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        result: dict = {}
        for loss_fn in self.loss_fns:
            loss_kind = getattr(loss_fn, "kind", "reconstruction")
            if loss_kind == "regularization":
                # Regularization loss (e.g., KL): pass distribution params
                val = loss_fn(mu=params["mu"], logvar=params["logvar"])
            elif loss_kind == "sae":
                # SAE losses
                x_hat_noisy_big = self.net.decode(params["v_noisy_big"])
                v_x_hat_noisy_big = self.net.encoder(x_hat_noisy_big) # type: ignore
                val = loss_fn(
                    x_hat_noisy_small=x_hat, x_hat_noisy_big=x_hat_noisy_big,
                    v_x_hat_noisy_big=v_x_hat_noisy_big, v_clean=params["v_clean"])
            else:
                # Reconstruction loss (e.g., L1, LPIPS): pass (x_hat, x)
                val = loss_fn(input=x_hat, target=x)
            result[loss_fn.id] = val.item()
            total_loss = total_loss + loss_fn.weight * val
        result["loss"] = total_loss
        result["x_hat"] = x_hat
        return result

register_type("Codec", Codec)

