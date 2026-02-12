from abc import ABC, abstractmethod
from typing import Optional, List

import torch

from fnn import *
from data import SaliencyAugmentation


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
        saliency_accuracy_threshold: float,
        saliency_warmup_threshold: float,
        saliency_blend_fraction: float = 0.0,
        saliency_label_smoothing: float = 0.0,
        saliency_augmentations: List[str] = [],
        saliency_learning_rate: Optional[float] = None,
        structure_pairing: bool = False,
        structure_candidates: int = 8,
        bf16: bool = False,
        codec: Config = {"net": {"type": "IdentityNet"}},
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
        self.saliency_blend_fraction = saliency_blend_fraction
        self.saliency_label_smoothing = saliency_label_smoothing
        self.saliency_augmentations = saliency_augmentations
        self.saliency_augment = SaliencyAugmentation(saliency_augmentations)
        self.saliency_learning_rate = saliency_learning_rate if saliency_learning_rate is not None else learning_rate
        self.structure_pairing = structure_pairing
        self.structure_candidates = structure_candidates
        self.bf16 = bf16
        # Codec: encode/decode between pixel and latent space
        self.codec: Codec = object_from_config(codec, type="Codec")
        latent_ch = self.codec.out_channels
        self.velocity_net = object_from_config(
            velocity_net,
            input_channels=latent_ch,
            output_channels=latent_ch,
        )
        self.num_downsamples = self.velocity_net.num_downsamples
        self.saliency_net: SaliencyNet = object_from_config(
            saliency_net,
            in_channels=latent_ch,
        )
        self.saliency_channels = self.saliency_net.output_channels
        # Generative mode flag (set by dataset during training)
        self.is_generative = False
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
            x: (B, C, H, W) input in latent space (or pixel space if IdentityNet)
            
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
        
        Images are encoded through the frozen codec before classification.
        The saliency network learns to distinguish domain 0 from domain 1
        in latent space.
        
        Args:
            x0: (B, 3, H, W) images from domain 0 (pixel space)
            x1: (B, 3, H, W) images from domain 1 (pixel space)
            
        Returns:
            dict with:
                - loss: scalar soft cross-entropy loss
                - accuracy: classification accuracy (argmax vs majority domain)
        """
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        # Encode through frozen codec to latent space
        with torch.no_grad():
            z0 = self.codec.encode(x0)  # (B, C, H', W')
            z1 = self.codec.encode(x1)  # (B, C, H', W')
        
        # Determine how many samples use blended interpolation vs pure domains
        num_blends = int(B * self.saliency_blend_fraction)
        num_pure = B - num_blends
        
        # Construct t values for all 2B samples:
        # - First num_pure samples from x0: t=0 (pure domain 0)
        # - First num_pure samples from x1: t=1 (pure domain 1)  
        # - Remaining num_blends from each: t sampled (blended)
        t_parts = []
        
        # Pure domain 0 samples (t=0)
        if num_pure > 0:
            t_parts.append(torch.zeros(num_pure, device=device, dtype=dtype))
        
        # Pure domain 1 samples (t=1)
        if num_pure > 0:
            t_parts.append(torch.ones(num_pure, device=device, dtype=dtype))
        
        # Blended samples (t sampled from model's timestep distribution)
        if num_blends > 0:
            # Sample t for blends applied to both x0 and x1 slices
            t_blends = self.sample_timestep(num_blends * 2, device, dtype)
            t_parts.append(t_blends)
        
        t = torch.cat(t_parts, dim=0)  # (2B,)
        
        # Construct input latents for interpolation z_t = t*z1 + (1-t)*z0
        z0_parts = []
        z1_parts = []
        
        if num_pure > 0:
            z0_parts.append(z0[:num_pure])
            z1_parts.append(z1[:num_pure])
            z0_parts.append(z0[:num_pure])
            z1_parts.append(z1[:num_pure])
        
        if num_blends > 0:
            z0_parts.append(z0[num_pure:])
            z0_parts.append(z0[num_pure:])
            z1_parts.append(z1[num_pure:])
            z1_parts.append(z1[num_pure:])
        
        z0_all = torch.cat(z0_parts, dim=0)  # (2B, C, H', W')
        z1_all = torch.cat(z1_parts, dim=0)  # (2B, C, H', W')
        
        # Compute interpolated latents: z_t = t*z1 + (1-t)*z0
        t_broadcast = t[:, None, None, None]  # (2B, 1, 1, 1)
        z_t = t_broadcast * z1_all + (1 - t_broadcast) * z0_all  # (2B, C, H', W')
        
        # Get classification logits from latent inputs
        logits = self.get_saliency(z_t)  # (2B, num_classes)
        
        # Soft targets: [1-t, t] for each sample
        # At t=0: [1, 0] = domain 0, at t=1: [0, 1] = domain 1
        soft_targets = torch.stack([1 - t, t], dim=1)  # (2B, 2)
        
        # Apply label smoothing: targets = targets * (1 - ε) + ε / num_classes
        # This prevents overconfidence and keeps gradients flowing
        if self.saliency_label_smoothing > 0:
            num_classes = soft_targets.shape[1]
            smooth_targets = soft_targets * (1 - self.saliency_label_smoothing)
            soft_targets = smooth_targets + self.saliency_label_smoothing / num_classes
        
        # Soft cross-entropy loss: -sum(targets * log_softmax(logits), dim=1)
        # This generalizes standard CE: at t=0 or t=1, it equals hard CE
        log_probs = F.log_softmax(logits, dim=1)  # (2B, 2)
        loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
        
        # Compute accuracy for monitoring: compare argmax to majority domain
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (2B,)
            labels = (t > 0.5).long()     # Majority domain
            accuracy = (preds == labels).float().mean()
        
        return {
            "loss": loss,
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
        Training forward pass with flow matching in latent space.
        
        Images are encoded through the frozen codec to latent space,
        then interpolated and velocity-predicted in that space.
        
        Args:
            x0: (B, 3, H, W) source domain image (pixel space)
            x1: (B, 3, H, W) target domain image (pixel space)
            t: (B,) timesteps in [0, 1]
            
        Returns:
            dict with:
                - v_pred: (B, C, H', W') predicted velocity in latent space
                - v_target: (B, C, H', W') target velocity in latent space
                - z_t: (B, C, H', W') interpolated latent state
        """
        # Encode to latent space (codec is frozen)
        with torch.no_grad():
            z0 = self.codec.encode(x0)  # (B, C, H', W')
            z1 = self.codec.encode(x1)  # (B, C, H', W')

        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_broadcast = t[:, None, None, None]
        
        # Linear interpolation in latent space: z_t = t * z1 + (1 - t) * z0
        z_t = t_broadcast * z1 + (1 - t_broadcast) * z0
        
        # Target velocity in latent space
        v_target = z1 - z0
        
        # Predict velocity at interpolated latent state
        v_pred = self.velocity_net(z_t, t)
        
        return {
            "v_pred": v_pred,
            "v_target": v_target,
            "z_t": z_t,
        }

    def compute_loss(self, x0: torch.Tensor, x1: torch.Tensor, t: Optional[torch.Tensor] = None) -> dict:
        """
        Compute saliency-weighted rectified flow matching loss in latent space.
        
        From the paper: min_v ∫ E[||∇h(z_t)^T * (z1 - z0 - v(z_t, t))||²] dt
        
        The saliency network h(z) operates on latent representations, and ∇h(z_t)^T
        acts as a saliency score that re-weights coordinates so the loss focuses on
        penalizing errors that cause significant changes in feature space.
        
        Args:
            x0: (B, 3, H, W) source domain image (pixel space)
            x1: (B, 3, H, W) target domain image (pixel space)
            t: (B,) timesteps in [0, 1], if None will be sampled
            
        Returns:
            dict with:
                - loss: scalar saliency-weighted MSE loss
                - v_pred: predicted velocity (latent space)
                - v_target: target velocity (latent space)
                - jvp_result: (B, saliency_channels) JVP result for analysis
        """
        from torch.autograd.functional import jvp
        
        B = x0.shape[0]
        
        if t is None:
            t = self.sample_timestep(B, x0.device, x0.dtype)
        
        out = self.forward(x0, x1, t)
        v_pred = out["v_pred"]
        v_target = out["v_target"]
        z_t = out["z_t"]
        
        # Velocity error in latent space: v_target - v_pred
        # NOTE: v_pred has gradients connected to velocity_net
        v_error = v_target - v_pred  # (B, C, H', W')
        
        if self.is_generative:
            # Generative mode: simple MSE loss (no saliency weighting)
            loss = (v_error ** 2).mean()
            jvp_result = None
        else:
            # Image translation mode: saliency-weighted loss in latent space
            # Compute: ||J_h(z_t) @ v_error||²
            
            def saliency_latent_fn(z):
                return self.saliency_net.get_latent(z)
            
            # JVP: J_h(z_t) @ v_error
            _, jvp_result = jvp(
                saliency_latent_fn,
                (z_t.detach(),),  # primal: detached interpolated latent
                (v_error,),       # tangent: velocity error (has gradients!)
                create_graph=True
            )
            
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
        
        Encodes to latent space, integrates velocity ODE, decodes back to pixels.
        
        Args:
            x0: (B, 3, H, W) input image (pixel space)
            num_steps: number of flow integration steps (default: self.num_inference_steps)
            
        Returns:
            x1: (B, 3, H, W) generated image (pixel space)
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        # Encode to latent space
        z = self.codec.encode(x0)  # (B, C, H', W')
        B = z.shape[0]

        # Euler integration in latent space
        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            t_val = step_idx * dt
            t = torch.full((B,), t_val, device=z.device, dtype=z.dtype)
            v = self.velocity_net(z, t)
            z = z + v * dt
        
        # Decode back to pixel space
        return self.codec.decode(z)

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


