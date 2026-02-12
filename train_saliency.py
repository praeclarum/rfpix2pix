import gc
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from fnn import save_module, device
from data import RFPix2pixDataset
from model import RFPix2pixModel
from utils import Colors as C, AccuracyTracker

SALIENCY_STATE_FILE = "saliency_state.json"


def read_saliency_state(run_dir: str) -> dict:
    """
    Read the saliency training state from the state file.
    
    Returns:
        dict with keys:
            - accuracy: float (0.0-1.0) or None if not yet trained
            - backbone_warmed_up: bool, True if backbone warmup phase is complete
    """
    state_path = os.path.join(run_dir, SALIENCY_STATE_FILE)
    default_state = {
        "accuracy": None,
        "backbone_warmed_up": False,
    }
    if not os.path.exists(state_path):
        return default_state
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
            for key, default_value in default_state.items():
                if key not in state:
                    state[key] = default_value
            return state
    except (ValueError, IOError):
        return default_state


def write_saliency_state(run_dir: str, accuracy: Optional[float] = None, backbone_warmed_up: Optional[bool] = None):
    """
    Update the saliency training state file.
    
    Only updates the fields that are provided (not None).
    """
    state = read_saliency_state(run_dir)
    if accuracy is not None:
        state["accuracy"] = accuracy
        print(f"{C.DIM}Saved saliency accuracy: {C.CYAN}{accuracy*100:.2f}%{C.RESET}")
    if backbone_warmed_up is not None:
        state["backbone_warmed_up"] = backbone_warmed_up
        print(f"{C.DIM}Saved backbone_warmed_up: {C.CYAN}{backbone_warmed_up}{C.RESET}")
    state_path = os.path.join(run_dir, SALIENCY_STATE_FILE)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def should_train_saliency(run_dir: str, threshold: float) -> bool:
    """
    Check if saliency training is needed.
    
    Args:
        run_dir: Run directory path
        threshold: Required accuracy (0-1)
        
    Returns:
        True if saliency training is needed, False otherwise.
    """
    state = read_saliency_state(run_dir)
    accuracy = state["accuracy"]
    if accuracy is None:
        print(f"{C.YELLOW}⚠ No saliency checkpoint found, training needed.{C.RESET}")
        return True
    if accuracy < threshold:
        print(f"{C.YELLOW}⚠ Saliency accuracy {accuracy*100:.2f}% < {threshold*100:.2f}% threshold, training needed.{C.RESET}")
        return True
    print(f"{C.GREEN}✓ Saliency accuracy {accuracy*100:.2f}% >= {threshold*100:.2f}% threshold, skipping saliency training.{C.RESET}")
    return False


def is_backbone_warmed_up(run_dir: str) -> bool:
    """Check if backbone warmup phase is complete."""
    state = read_saliency_state(run_dir)
    return state["backbone_warmed_up"]


def compute_saliency_loss(
    codec,
    saliency,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> dict:
    """
    Compute classification loss for training the saliency network.
    
    Images are encoded through the frozen codec before classification.
    The saliency network learns to distinguish domain 0 from domain 1
    in latent space.
    
    Args:
        codec: Codec module (frozen)
        saliency: Saliency module (trainable)
        x0: (B, 3, H, W) images from domain 0 (pixel space)
        x1: (B, 3, H, W) images from domain 1 (pixel space)
        
    Returns:
        dict with:
            - loss: scalar soft cross-entropy loss
            - accuracy: classification accuracy (argmax vs majority domain)
    """
    B = x0.shape[0]
    dev = x0.device
    dtype = x0.dtype

    # Encode through frozen codec to latent space
    with torch.no_grad():
        z0 = codec.encode(x0)  # (B, C, H', W')
        z1 = codec.encode(x1)  # (B, C, H', W')
    
    # Determine how many samples use blended interpolation vs pure domains
    num_blends = int(B * saliency.blend_fraction)
    num_pure = B - num_blends
    
    # Construct t values for all 2B samples:
    # - First num_pure samples from x0: t=0 (pure domain 0)
    # - First num_pure samples from x1: t=1 (pure domain 1)  
    # - Remaining num_blends from each: t sampled (blended)
    t_parts = []
    
    # Pure domain 0 samples (t=0)
    if num_pure > 0:
        t_parts.append(torch.zeros(num_pure, device=dev, dtype=dtype))
    
    # Pure domain 1 samples (t=1)
    if num_pure > 0:
        t_parts.append(torch.ones(num_pure, device=dev, dtype=dtype))
    
    # Blended samples (t sampled from velocity's timestep distribution)
    if num_blends > 0:
        # Use uniform sampling for saliency blend (simple and sufficient)
        t_blends = torch.rand(num_blends * 2, device=dev, dtype=dtype)
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
    logits = saliency(z_t)  # (2B, num_classes)
    
    # Soft targets: [1-t, t] for each sample
    # At t=0: [1, 0] = domain 0, at t=1: [0, 1] = domain 1
    soft_targets = torch.stack([1 - t, t], dim=1)  # (2B, 2)
    
    # Apply label smoothing: targets = targets * (1 - ε) + ε / num_classes
    if saliency.label_smoothing > 0:
        num_classes = soft_targets.shape[1]
        smooth_targets = soft_targets * (1 - saliency.label_smoothing)
        soft_targets = smooth_targets + saliency.label_smoothing / num_classes
    
    # Soft cross-entropy loss: -sum(targets * log_softmax(logits), dim=1)
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


def train_saliency(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, dev: bool) -> float:
    """
    Train the saliency network until it reaches acceptable accuracy.
    
    Returns:
        Final accuracy as a float (0.0 to 1.0)
    """
    # Ensure codec is frozen, saliency is trainable
    rf_model.eval_codec()
    rf_model.train_saliency()
    rf_model.to(device)

    saliency = rf_model.saliency
    codec = rf_model.codec

    batch_size = saliency.net.__config.get("train_batch_size", 48) if hasattr(saliency.net, "__config") else 48
    # Use velocity's batch sizes as defaults since saliency doesn't own them
    vel = rf_model.velocity
    batch_size = vel.train_batch_size
    minibatch_size = vel.train_minibatch_size
    num_steps = vel.train_images // batch_size
    last_step = num_steps
    
    lr = saliency.learning_rate
    
    # Check if backbone warmup was already completed (resuming training)
    backbone_already_warmed = is_backbone_warmed_up(run_dir)
    
    def create_optimizer(backbone_frozen: bool) -> torch.optim.AdamW:
        """Create optimizer with appropriate parameter groups."""
        param_groups = saliency.get_optimizer_param_groups(lr, backbone_frozen)
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-4)
    
    if backbone_already_warmed:
        # Resume with unfrozen backbone
        saliency.unfreeze_backbone()
        backbone_frozen = False
    else:
        # Start with frozen backbone for warmup
        saliency.freeze_backbone()
        backbone_frozen = True
    
    optimizer = create_optimizer(backbone_frozen)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True, num_workers=8)
    data_iter = iter(dataloader)

    run_id = os.path.basename(run_dir)

    def save(step: int):
        save_module(saliency, os.path.join(run_dir, f"saliency_{run_id}_{step:06d}.ckpt"))

    num_grad_acc_steps = max(1, batch_size // minibatch_size)
    grad_scale = 1.0 / num_grad_acc_steps
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Training Saliency Network ━━━{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Steps:{C.RESET}          {C.BOLD}{num_steps}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {minibatch_size}")
    print(f"{C.BRIGHT_CYAN}  Grad acc steps:{C.RESET} {num_grad_acc_steps}")
    print(f"{C.BRIGHT_CYAN}  Learning rate:{C.RESET}  {lr}")
    print(f"{C.BRIGHT_CYAN}  Backbone frozen:{C.RESET} {C.YELLOW if backbone_frozen else C.GREEN}{backbone_frozen}{C.RESET}\n")

    save_steps = 1024
    next_save_step = save_steps

    accuracy_item = 0.0
    accuracy_tracker = AccuracyTracker()

    progress = tqdm(range(0, last_step), initial=0, total=last_step)
    for step in progress:
        optimizer.zero_grad()
        
        loss_item = 0.0
        accuracy_item = 0.0
        
        for grad_step in range(num_grad_acc_steps):
            # Get data batch
            try:
                inputs = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs = next(data_iter)
            
            input_domain_0 = inputs["domain_0"].to(device)  # (B, 3, H, W)
            input_domain_1 = inputs["domain_1"].to(device)  # (B, 3, H, W)
            
            # Apply saliency-specific augmentations (if configured)
            input_domain_0 = saliency.augment(input_domain_0)
            input_domain_1 = saliency.augment(input_domain_1)
            
            output = compute_saliency_loss(codec, saliency, input_domain_0, input_domain_1)
            loss = output['loss']
            grad_loss: torch.Tensor = loss * grad_scale
            grad_loss.backward()

            accuracy_item += output['accuracy'].item() * grad_scale
            loss_item += grad_loss.item()

        optimizer.step()
        gc.collect()
        
        # Update accuracy tracker with current measurement
        accuracy_tracker.update(accuracy_item)
        smoothed_accuracy = accuracy_tracker.smoothed

        progress.set_postfix({
            "loss": loss_item,
            "acc": f"{accuracy_item*100:.2f}%",
            "acc_smooth": f"{smoothed_accuracy*100:.2f}%"
        })
        
        if step >= next_save_step:
            save(step)
            next_save_step = step + save_steps

        # Use smoothed accuracy for phase transition checks
        if backbone_frozen and accuracy_tracker.above_threshold(saliency.warmup_threshold):
            print(f"\n{C.BOLD}{C.GREEN}✓ Backbone warmup complete{C.RESET} at step {C.CYAN}{step}{C.RESET}, smoothed accuracy {C.BRIGHT_GREEN}{smoothed_accuracy*100:.2f}%{C.RESET}")
            saliency.unfreeze_backbone()
            # Create new optimizer with differential learning rates
            optimizer = create_optimizer(backbone_frozen=False)
            backbone_frozen = False
            # Reset tracker when entering new phase
            accuracy_tracker.reset()
            save(step)
            write_saliency_state(run_dir, accuracy=smoothed_accuracy, backbone_warmed_up=True)
        elif not backbone_frozen and accuracy_tracker.above_threshold(saliency.accuracy_threshold):
            print(f"\n{C.BOLD}{C.BRIGHT_GREEN}✓ Saliency training complete{C.RESET} at step {C.CYAN}{step}{C.RESET}, smoothed accuracy {C.BRIGHT_GREEN}{smoothed_accuracy*100:.2f}%{C.RESET}\n")
            save(step)
            return accuracy_tracker.smoothed
        
    # Training ended without reaching threshold - return smoothed accuracy if available
    save(last_step)
    return accuracy_tracker.smoothed if accuracy_tracker.is_stable else accuracy_item
