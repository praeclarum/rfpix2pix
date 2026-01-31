from abc import ABC, abstractmethod
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
        saliency_accuracy_threshold: int,
        saliency_warmup_threshold: int = 70,
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
        self.saliency_accuracy_threshold = saliency_accuracy_threshold
        self.saliency_warmup_threshold = saliency_warmup_threshold
        self.velocity_net = object_from_config(
            velocity_net,
            input_channels=3,
            output_channels=3,
        )
        self.num_downsamples = self.velocity_net.num_downsamples
        self.saliency_net: SaliencyNet = object_from_config(saliency_net)
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

class SaliencyNet(nn.Module, ABC):
    """
    Abstract base class for saliency networks used in style transfer.
    
    Saliency networks provide a feature mapping h(x) for saliency-weighted loss.
    The gradient ∇h(x) acts as a saliency score that re-weights coordinates
    so the loss focuses on penalizing errors that cause significant changes
    in feature space.
    
    Subclasses must implement:
        - forward(x): Returns classification logits for domain classification
        - get_latent(x): Returns latent feature representation h(x)
        - output_channels: Number of channels in the latent representation
    
    Subclasses may optionally implement:
        - freeze_backbone(): Freeze pretrained weights for head warmup
        - unfreeze_backbone(): Unfreeze for full fine-tuning
    """
    
    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of channels in the latent feature representation."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for domain classification.
        
        Args:
            x: (B, 3, H, W) input images in [-1, 1] range
            
        Returns:
            (B, num_classes) classification logits
        """
        pass
    
    @abstractmethod
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent feature representation h(x).
        
        This is the feature mapping used for saliency-weighted loss.
        The gradient of this function provides the saliency weights.
        
        Args:
            x: (B, 3, H, W) input images in [-1, 1] range
            
        Returns:
            (B, output_channels) latent features
        """
        pass
    
    def freeze_backbone(self):
        """
        Freeze pretrained backbone weights for head warmup training.
        
        Default implementation does nothing. Override in subclasses
        that use pretrained backbones.
        """
        pass
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone for full fine-tuning.
        
        Default implementation does nothing. Override in subclasses
        that use pretrained backbones.
        """
        pass
    
    def get_optimizer_param_groups(self, lr: float, backbone_frozen: bool) -> list:
        """
        Get parameter groups for optimizer with appropriate learning rates.
        
        Override this method to provide differential learning rates for
        pretrained backbones vs new head layers.
        
        Args:
            lr: Base learning rate
            backbone_frozen: Whether backbone is currently frozen
            
        Returns:
            List of parameter group dicts for optimizer, e.g.:
            [{'params': [...], 'lr': lr}, {'params': [...], 'lr': lr * 0.1}]
        """
        # Default: single group with all parameters at base LR
        return [{'params': self.parameters(), 'lr': lr}]


class ResNetSaliencyNet(SaliencyNet):
    """
    ResNet-based saliency network for domain classification.
    
    Uses a pretrained ResNet backbone from torchvision to extract features,
    then classifies whether an image belongs to domain 0 or domain 1.
    
    Handles colorspace conversion from app's [-1, 1] range to ImageNet
    normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    """
    
    # ResNet backbone output channels for each variant
    BACKBONE_CHANNELS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }
    
    # Type hints for registered buffers
    imagenet_mean: torch.Tensor
    imagenet_std: torch.Tensor
    
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        latent_channels: int = 512,
    ):
        """
        Initialize ResNet saliency network.
        
        Args:
            backbone: ResNet variant to use ("resnet18", "resnet34", "resnet50", 
                      "resnet101", "resnet152")
            num_classes: Number of domain classes (typically 2)
            pretrained: Whether to use ImageNet pretrained weights
            latent_channels: Dimension of the latent feature space h(x)
        """
        super().__init__()
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self._output_channels = latent_channels
        
        if backbone not in self.BACKBONE_CHANNELS:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Supported: {list(self.BACKBONE_CHANNELS.keys())}"
            )
        
        # Register ImageNet normalization constants as buffers (move with model)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        # Load pretrained backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        backbone_out_channels = self.BACKBONE_CHANNELS[backbone]
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Latent projection (this is h(x))
        self.latent_proj = nn.Linear(backbone_out_channels, latent_channels)
        
        # Classification head
        self.classifier = nn.Linear(latent_channels, num_classes)
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create ResNet backbone without the final FC layer."""
        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
        elif backbone == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = resnet34(weights=weights)
        elif backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
        elif backbone == "resnet101":
            from torchvision.models import resnet101, ResNet101_Weights
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            base = resnet101(weights=weights)
        elif backbone == "resnet152":
            from torchvision.models import resnet152, ResNet152_Weights
            weights = ResNet152_Weights.DEFAULT if pretrained else None
            base = resnet152(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
    
    def freeze_backbone(self):
        """
        Freeze the pretrained backbone weights.
        
        Call this before initial training to let the new head layers
        (latent_proj and classifier) learn meaningful representations
        before fine-tuning the backbone.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def unfreeze_backbone(self):
        """
        Unfreeze the backbone for full fine-tuning.
        
        Call this after the head layers have been warmed up to allow
        end-to-end training of the entire network.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
    
    def get_optimizer_param_groups(self, lr: float, backbone_frozen: bool) -> list:
        """
        Get parameter groups with differential learning rates.
        
        When backbone is frozen, only head layers are included.
        When backbone is unfrozen, backbone gets 10x lower LR to preserve
        pretrained features.
        
        Args:
            lr: Base learning rate for head layers
            backbone_frozen: Whether backbone is currently frozen
            
        Returns:
            List of parameter group dicts for optimizer
        """
        if backbone_frozen:
            # Only head layers train during warmup
            return [
                {'params': self.latent_proj.parameters(), 'lr': lr},
                {'params': self.classifier.parameters(), 'lr': lr},
            ]
        else:
            # Differential LR: backbone gets 10x lower LR
            backbone_lr = lr * 0.1
            return [
                {'params': self.backbone.parameters(), 'lr': backbone_lr},
                {'params': self.latent_proj.parameters(), 'lr': lr},
                {'params': self.classifier.parameters(), 'lr': lr},
            ]
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from app's [-1, 1] colorspace to ImageNet normalization.
        
        App colorspace: [-1, 1]
        ImageNet expects: (x - mean) / std where x is in [0, 1]
        
        Conversion:
            1. [-1, 1] -> [0, 1]: x_01 = (x + 1) / 2
            2. [0, 1] -> ImageNet: (x_01 - mean) / std
        """
        # Convert [-1, 1] to [0, 1]
        x_01 = (x + 1.0) / 2.0
        # Apply ImageNet normalization
        x_normalized = (x_01 - self.imagenet_mean) / self.imagenet_std
        return x_normalized
    
    @property
    def output_channels(self) -> int:
        """Number of channels in the latent feature representation."""
        return self._output_channels
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent feature representation h(x).
        
        Args:
            x: (B, 3, H, W) input images in [-1, 1] range
            
        Returns:
            (B, latent_channels) latent features
        """
        x_normalized = self._normalize_input(x)
        features = self.backbone(x_normalized)  # (B, C, H', W')
        pooled = self.pool(features)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)    # (B, C)
        latent = self.latent_proj(pooled)  # (B, latent_channels)
        return latent
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for domain classification.
        
        Args:
            x: (B, 3, H, W) input images in [-1, 1] range
            
        Returns:
            (B, num_classes) classification logits
        """
        latent = self.get_latent(x)
        logits = self.classifier(latent)
        return logits

register_type("ResNetSaliencyNet", ResNetSaliencyNet)
