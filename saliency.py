from abc import ABC, abstractmethod
from typing import Optional, List

import torch

from fnn import *
from data import SaliencyAugmentation


class Saliency(nn.Module):
    """
    Saliency module holder: wraps a SaliencyNet with training parameters
    and data augmentation.
    
    The net config determines the actual saliency network architecture.
    Training parameters control the saliency training phase.
    
    Example config:
        {
            "type": "Saliency",
            "net": {"type": "ResNetSaliencyNet", "backbone": "resnet34", ...},
            "learning_rate": 0.00005,
            "accuracy_threshold": 0.995,
            "warmup_threshold": 0.90,
            "blend_fraction": 0.0,
            "label_smoothing": 0.1,
            "augmentations": ["color_jitter", "grayscale", "hflip", "random_erasing"]
        }
    """
    def __init__(
        self,
        net: Config,
        in_channels: int = 3,
        learning_rate: float = 0.0001,
        accuracy_threshold: float = 0.995,
        warmup_threshold: float = 0.90,
        blend_fraction: float = 0.0,
        label_smoothing: float = 0.0,
        augmentations: list[str] = [],
    ):
        super().__init__()
        self.net: SaliencyNet = object_from_config(net, in_channels=in_channels)
        self.learning_rate = learning_rate
        self.accuracy_threshold = accuracy_threshold
        self.warmup_threshold = warmup_threshold
        self.blend_fraction = blend_fraction
        self.label_smoothing = label_smoothing
        self.augmentations = augmentations
        self.augment = SaliencyAugmentation(augmentations)

    @property
    def output_channels(self) -> int:
        """Number of channels in the latent feature representation."""
        return self.net.output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for domain classification."""
        return self.net(x)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent feature representation h(x)."""
        return self.net.get_latent(x)

    def freeze_backbone(self):
        """Freeze pretrained backbone weights."""
        self.net.freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        self.net.unfreeze_backbone()

    def get_optimizer_param_groups(self, lr: float, backbone_frozen: bool) -> list:
        """Get parameter groups with appropriate learning rates."""
        return self.net.get_optimizer_param_groups(lr, backbone_frozen)
register_type("Saliency", Saliency)

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
    
    When in_channels=3 (pixel space): uses full pretrained backbone with
    ImageNet normalization.
    
    When in_channels!=3 (latent space): replaces the aggressive initial
    downsampling (conv1 stride=2 + maxpool stride=2) with a gentle
    Conv2d(in_channels, 64, 3, stride=1, padding=1). Skips ImageNet
    normalization since latents aren't in that distribution.
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
        in_channels: int = 3,
    ):
        """
        Initialize ResNet saliency network.
        
        Args:
            backbone: ResNet variant to use
            num_classes: Number of domain classes (typically 2)
            pretrained: Whether to use ImageNet pretrained weights
            latent_channels: Dimension of the latent feature space h(x)
            in_channels: Number of input channels (3 for pixel space,
                        codec.out_channels for latent space)
        """
        super().__init__()
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self._output_channels = latent_channels
        self.in_channels = in_channels
        self._use_imagenet_norm = (in_channels == 3)
        
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
        self.backbone = self._create_backbone(backbone, pretrained, in_channels)
        backbone_out_channels = self.BACKBONE_CHANNELS[backbone]
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Latent projection (this is h(x))
        self.latent_proj = nn.Linear(backbone_out_channels, latent_channels)
        
        # Classification head
        self.classifier = nn.Linear(latent_channels, num_classes)
    
    def _create_backbone(self, backbone: str, pretrained: bool, in_channels: int) -> nn.Module:
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
        
        if in_channels == 3:
            # Standard pixel-space path: keep pretrained conv1+bn1+relu+maxpool
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
        else:
            # Latent-space path: replace conv1(7x7,stride=2)+maxpool(stride=2)
            # with a gentle Conv2d(in_channels, 64, 3, stride=1, padding=1)
            # to avoid aggressive downsampling on already-small latent maps.
            input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(input_conv.weight, mode='fan_out', nonlinearity='relu')
            return nn.Sequential(
                input_conv,
                base.bn1,
                base.relu,
                # No maxpool — preserve spatial resolution
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
        Normalize input for the backbone.
        
        For pixel-space inputs (in_channels=3): converts [-1, 1] to ImageNet normalization.
        For latent-space inputs: passes through unchanged.
        """
        if not self._use_imagenet_norm:
            return x
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
            x: (B, C, H, W) input (pixel or latent space depending on in_channels)
            
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
            x: (B, C, H, W) input (pixel or latent space depending on in_channels)
            
        Returns:
            (B, num_classes) classification logits
        """
        latent = self.get_latent(x)
        logits = self.classifier(latent)
        return logits

register_type("ResNetSaliencyNet", ResNetSaliencyNet)
