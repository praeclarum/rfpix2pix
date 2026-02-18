import gc
import math
import random
import copy
import datetime
import os
from contextlib import nullcontext, contextmanager
from typing import Optional

import torch
from torch.autograd.functional import jvp
from tqdm import tqdm
import wandb
import wandb.wandb_run
from PIL import Image

from fnn import save_module, device
from data import RFPix2pixDataset, StructurePairing, load_media_item
from model import RFPix2pixModel
from utils import Colors as C


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


def compute_velocity_loss(
    codec,
    velocity,
    saliency,
    x0: torch.Tensor,
    x1: torch.Tensor,
    is_generative: bool,
    t: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute saliency-weighted rectified flow matching loss in latent space.
    
    From the paper: min_v ∫ E[||∇h(z_t)^T * (z1 - z0 - v(z_t, t))||²] dt
    
    Args:
        codec: Codec module (frozen)
        velocity: Velocity module (trainable)
        saliency: Saliency module (frozen)
        x0: (B, 3, H, W) source domain image (pixel space)
        x1: (B, 3, H, W) target domain image (pixel space)
        is_generative: if True, use simple MSE loss instead of saliency-weighted
        t: (B,) timesteps in [0, 1], if None will be sampled
        
    Returns:
        dict with:
            - loss: scalar loss
            - v_pred: predicted velocity (latent space)
            - v_target: target velocity (latent space)
            - jvp_result: JVP result (or None in generative mode)
    """
    B = x0.shape[0]
    
    if t is None:
        t = velocity.sample_timestep(B, x0.device, x0.dtype)
    
    # Encode to latent space (codec is frozen)
    with torch.no_grad():
        z1 = codec.encode(x1)  # (B, C, H', W')
        if is_generative:
            # Generative mode: draw z0 directly in latent space (standard normal)
            # Do NOT encode pixel-space noise through the codec
            z0 = torch.randn_like(z1)
        else:
            z0 = codec.encode(x0)  # (B, C, H', W')

    # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
    t_broadcast = t[:, None, None, None] # type: ignore
    
    # Linear interpolation in latent space: z_t = t * z1 + (1 - t) * z0
    z_t = t_broadcast * z1 + (1 - t_broadcast) * z0
    
    # Target velocity in latent space
    v_target = z1 - z0
    
    # Predict velocity at interpolated latent state
    v_pred = velocity(z_t, t)
    
    # Velocity error in latent space
    v_error = v_target - v_pred  # (B, C, H', W')
    
    if is_generative:
        # Generative mode: simple MSE loss (no saliency weighting)
        loss = (v_error ** 2).mean()
        jvp_result = None
    else:
        # Image translation mode: saliency-weighted loss in latent space
        # Compute: ||J_h(z_t) @ v_error||²
        
        def saliency_latent_fn(z):
            return saliency.get_latent(z)
        
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
def sample(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, name: str, ema: Optional[EMA] = None):
    """Generate sample grid showing [input | output] pairs. Uses EMA weights if provided."""
    rows = []
    is_generative = dataset.use_random_noise_domain0
    num_samples = rf_model.velocity.sample_batch_size
    sf = rf_model.codec.spatial_factor
    latent_ch = rf_model.codec.out_channels
    h = dataset.max_image_size // sf
    w = h
    
    # Use EMA weights for inference if available
    ema_ctx = ema.swap(rf_model.velocity.net) if ema is not None else nullcontext()
    with ema_ctx:
        for i in range(num_samples):
            if is_generative:
                z0 = torch.randn(1, latent_ch, h, w, device=device)
            else:
                inputs = dataset[random.randint(0, len(dataset) - 1)]
                input_image = inputs["domain_0"].unsqueeze(0).to(device)  # (1, 3, H, W)
                z0 = rf_model.codec.encode(input_image)
            output_image = rf_model.generate(z0)
            # Show z0 input (first 3 channels) alongside output
            input_vis = z0[:, :3, :, :]
            # Upsample input_vis to match output spatial size
            if input_vis.shape[2:] != output_image.shape[2:]:
                input_vis = torch.nn.functional.interpolate(input_vis, size=output_image.shape[2:], mode='nearest')
            row = torch.cat([input_vis, output_image], dim=3)  # (1, 3, H, 2W)
            rows.append(row)
        image = torch.cat(rows, dim=2)  # (1, 3, H*B, 2W)
        image = (image.squeeze(0).cpu().numpy() + 1.0) * 127.5
        image = image.clip(0, 255).astype("uint8")
        image = Image.fromarray(image.transpose(1, 2, 0))
        path = os.path.join(run_dir, f"{name}.jpg")
        image.save(path, quality=90)
    print(f"Saved sample grid to {path} ({image.width}x{image.height})")


@torch.no_grad()
def sample_structure_pairings(
    dataset: RFPix2pixDataset,
    pairing: StructurePairing,
    run_dir: str,
    num_samples: int = 16,
):
    """
    Generate a proof sheet showing structure pairings for debugging.
    
    Creates a grid where each row shows:
    - Column 0: A domain 0 media frame
    - Columns 1..K: Top-K most similar domain 1 media frames (sorted by similarity)
    """
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Structure Pairing Proof Sheet ━━━{C.RESET}")
    
    n_domain_0 = len(dataset.domain_0_media_items)
    k = pairing.top_k_indices.shape[1]
    
    if num_samples >= n_domain_0:
        domain_0_indices = list(range(n_domain_0))
    else:
        domain_0_indices = random.sample(range(n_domain_0), num_samples)
    
    rows = []
    for d0_idx in tqdm(domain_0_indices, desc="Building proof sheet"):
        d0_item = dataset.domain_0_media_items[d0_idx]
        d0_image = load_media_item(
            d0_item,
            dataset.max_image_size,
        )
        
        d1_indices = pairing.top_k_indices[d0_idx]
        
        row_images = [d0_image]
        for d1_idx in d1_indices:
            d1_item = dataset.domain_1_media_items[d1_idx]
            d1_image = load_media_item(
                d1_item,
                dataset.max_image_size,
            )
            row_images.append(d1_image)
        
        row = torch.cat([img.unsqueeze(0) for img in row_images], dim=3)
        rows.append(row)
    
    grid = torch.cat(rows, dim=2)
    grid = (grid.squeeze(0).cpu().numpy() + 1.0) * 127.5
    grid = grid.clip(0, 255).astype("uint8")
    grid = Image.fromarray(grid.transpose(1, 2, 0))
    
    path = os.path.join(run_dir, "pairings.jpg")
    grid.save(path, quality=90)
    print(f"{C.GREEN}✓ Saved structure pairing proof sheet to {C.BOLD}{path}{C.RESET}")
    print(f"  {C.DIM}Grid: {len(domain_0_indices)} rows × {k + 1} columns (source + {k} matches){C.RESET}\n")


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, schedule: str) -> float:
    """Compute learning rate for a given step with warmup + schedule."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return max_lr
    # Cosine decay from max_lr -> 0 over remaining steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_velocity(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, step_start: int, dev: bool):
    """
    Train the velocity network using saliency-weighted flow matching loss.
    
    The codec and saliency network are frozen. The velocity network is trained
    to predict the flow velocity in latent space.
    
    Supports:
        - Cosine LR schedule with linear warmup (lr_schedule, warmup_images)
        - Gradient norm clipping (gradient_clip)
        - Exponential Moving Average of weights for sampling (ema)
    
    Args:
        rf_model: The RFPix2pixModel with trained codec and saliency
        dataset: Dataset providing paired domain images
        run_dir: Directory for saving checkpoints
        step_start: Step to resume from (0 for fresh start)
        dev: If True, skip wandb logging
    """
    # Ensure codec and saliency are frozen, velocity is trainable
    rf_model.eval_codec()
    rf_model.eval_saliency()
    rf_model.velocity.net.train()
    rf_model.to(device)

    velocity = rf_model.velocity
    is_generative = dataset.use_random_noise_domain0

    num_steps = velocity.train_images // velocity.train_batch_size
    last_step = step_start + num_steps
    
    max_lr = velocity.learning_rate
    warmup_steps = velocity.warmup_images // velocity.train_batch_size
    
    optimizer = torch.optim.AdamW(
        velocity.net.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # EMA setup
    ema_tracker: Optional[EMA] = None
    if velocity.ema > 0:
        ema_tracker = EMA(velocity.net, decay=velocity.ema)
        # Load EMA state if resuming
        ema_path = os.path.join(run_dir, "ema_state.pt")
        if os.path.exists(ema_path):
            ema_tracker.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))
            print(f"{C.GREEN}✓ Loaded EMA state from {ema_path}{C.RESET}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=velocity.train_minibatch_size,
        shuffle=True,
        num_workers=8
    )
    data_iter = iter(dataloader)

    run_id = os.path.basename(run_dir)

    if dev:
        wandb_run: Optional[wandb.wandb_run.Run] = None
    else:
        wandb_run: Optional[wandb.wandb_run.Run] = wandb.init(
            project="rfpix2pix_gen" if is_generative else "rfpix2pix",
            save_code=True,
            id=run_id,
            config=rf_model.__config,  # pyright: ignore[reportArgumentType]
        )
        wandb_run.watch(velocity.net)

    def save(step: int):
        save_module(velocity, os.path.join(run_dir, f"velocity_{run_id}_{step:06d}.ckpt"))
        if ema_tracker is not None:
            torch.save(ema_tracker.state_dict(), os.path.join(run_dir, "ema_state.pt"))

    num_grad_acc_steps = max(1, velocity.train_batch_size // velocity.train_minibatch_size)
    grad_scale = 1.0 / num_grad_acc_steps
    
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Training Velocity Network ━━━{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Steps:{C.RESET}          {C.BOLD}{num_steps}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Starting step:{C.RESET}  {step_start}")
    print(f"{C.BRIGHT_CYAN}  Batch size:{C.RESET}     {velocity.train_batch_size}")
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {velocity.train_minibatch_size}")
    print(f"{C.BRIGHT_CYAN}  Grad acc steps:{C.RESET} {num_grad_acc_steps}")
    print(f"{C.BRIGHT_CYAN}  Learning rate:{C.RESET}  {max_lr}")
    print(f"{C.BRIGHT_CYAN}  LR schedule:{C.RESET}    {velocity.lr_schedule}" + (f" (warmup: {warmup_steps} steps / {velocity.warmup_images} images)" if warmup_steps > 0 else ""))
    if velocity.gradient_clip > 0:
        print(f"{C.BRIGHT_CYAN}  Gradient clip:{C.RESET}  {velocity.gradient_clip}")
    if ema_tracker is not None:
        print(f"{C.BRIGHT_CYAN}  EMA decay:{C.RESET}      {velocity.ema}")
    print()

    sample_minutes = 1
    max_sample_minutes = 8
    last_sample_time = datetime.datetime.now()

    save_minutes = 30
    last_save_time = datetime.datetime.now()

    progress = tqdm(range(step_start, last_step), initial=step_start, total=last_step - step_start)
    for step in progress:
        # Update learning rate
        lr = get_lr(step - step_start, warmup_steps, num_steps, max_lr, velocity.lr_schedule)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        
        loss_item = 0.0
        
        for grad_step in range(num_grad_acc_steps):
            # Get data batch
            try:
                inputs = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs = next(data_iter)
            
            input_domain_0 = inputs["domain_0"].to(device)  # (B, 3, H, W)
            input_domain_1 = inputs["domain_1"].to(device)  # (B, 3, H, W)
            
            # Compute saliency-weighted velocity loss
            autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if velocity.bf16 else nullcontext() # pyright: ignore[reportPrivateImportUsage]
            with autocast_ctx:
                output = compute_velocity_loss(
                    rf_model.codec, velocity, rf_model.saliency,
                    input_domain_0, input_domain_1,
                    is_generative=is_generative,
                )
                loss = output['loss']
            grad_loss: torch.Tensor = loss * grad_scale
            grad_loss.backward()

            loss_item += grad_loss.item()

        # Gradient clipping (after all accumulation, before optimizer step)
        if velocity.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(velocity.net.parameters(), max_norm=velocity.gradient_clip)

        optimizer.step()

        # EMA update (after optimizer step)
        if ema_tracker is not None:
            ema_tracker.update(velocity.net)

        gc.collect()
        
        progress.set_postfix({
            "loss": f"{loss_item:.4f}",
            "lr": f"{lr:.2e}",
        })
        
        if wandb_run is not None:
            wlog = {
                "vel_loss": loss_item,
                "lr": lr,
                "step": step,
            }
            wandb_run.log(wlog)

        if (datetime.datetime.now() - last_sample_time).total_seconds() >= sample_minutes * 60:
            sample(rf_model, dataset, run_dir, f"step_{step:06d}", ema=ema_tracker)
            sample_minutes = min(sample_minutes * 2, max_sample_minutes)
            last_sample_time = datetime.datetime.now()
        
        if (step == 0) or (datetime.datetime.now() - last_save_time).total_seconds() >= save_minutes * 60:
            save(step)
            last_save_time = datetime.datetime.now()

    # Final save (real weights for potential resume)
    save(last_step)
    # Save EMA weights as the final inference checkpoint
    if ema_tracker is not None:
        with ema_tracker.swap(velocity.net):
            save_module(velocity, os.path.join(run_dir, f"velocity_{run_id}_final_ema.ckpt"))
        print(f"{C.GREEN}✓ Saved EMA weights to velocity_{run_id}_final_ema.ckpt{C.RESET}")
    print(f"\n{C.BOLD}{C.BRIGHT_GREEN}✓ Velocity training complete{C.RESET} at step {C.CYAN}{last_step}{C.RESET}\n")
