import gc
import random
import datetime
import json
import os
from typing import Optional

import torch
from tqdm import tqdm
import wandb
import wandb.wandb_run
from PIL import Image

from fnn import save_module, device, get_lr
from data import RFPix2pixDataset
from model import RFPix2pixModel
from codec import Codec
from utils import Colors as C, AccuracyTracker

CODEC_STATE_FILE = "codec_state.json"


def read_codec_state(run_dir: str) -> dict:
    """Read the codec training state from the state file."""
    state_path = os.path.join(run_dir, CODEC_STATE_FILE)
    default_state = {
        "trained": False,
        "loss": None,
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


def write_codec_state(run_dir: str, trained: bool | None = None, loss: float | None = None):
    """Update the codec training state file."""
    state = read_codec_state(run_dir)
    if trained is not None:
        state["trained"] = trained
    if loss is not None:
        state["loss"] = loss
        print(f"{C.DIM}Saved codec loss: {C.CYAN}{loss:.6f}{C.RESET}")
    state_path = os.path.join(run_dir, CODEC_STATE_FILE)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def should_train_codec(rf_model: RFPix2pixModel, run_dir: str) -> bool:
    """Check if codec training is needed."""
    if rf_model.codec.train_images <= 0:
        return False
    state = read_codec_state(run_dir)
    if state["trained"]:
        print(f"{C.GREEN}✓ Codec already trained (loss={state['loss']:.6f}), skipping.{C.RESET}")
        return False
    print(f"{C.YELLOW}⚠ Codec training needed ({rf_model.codec.train_images} images).{C.RESET}")
    return True


def train_codec(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, dev: bool) -> float:
    """
    Train the codec (encoder/decoder) on both domains.
    
    Trains using the configured losses (e.g., L1 + LPIPS + KL for a VAE).
    Supports backbone warmup for pretrained encoder nets.
    
    Returns:
        Final loss value.
    """
    rf_model.train_codec()
    rf_model.eval_saliency()
    rf_model.to(device)

    codec: Codec = rf_model.codec
    batch_size = codec.train_batch_size if codec.train_batch_size is not None else 64
    minibatch_size = codec.train_minibatch_size if codec.train_minibatch_size is not None else batch_size

    num_steps = codec.train_images // batch_size
    last_step = num_steps

    max_lr = codec.learning_rate
    warmup_images = int(codec.warmup_fraction * codec.train_images)
    warmup_steps = warmup_images // codec.train_batch_size

    optimizer = torch.optim.AdamW(
        codec.net.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=minibatch_size, shuffle=True, num_workers=8
    )
    data_iter = iter(dataloader)

    run_id = os.path.basename(run_dir)

    if dev:
        wandb_run: Optional[wandb.wandb_run.Run] = None
    else:
        wandb_run: Optional[wandb.wandb_run.Run] = wandb.init(
            project="rfpix2pix_codec",
            save_code=True,
            id=run_id,
            config=rf_model.__config,  # pyright: ignore[reportArgumentType]
        )

    num_grad_acc_steps = max(1, batch_size // minibatch_size)
    grad_scale = 1.0 / num_grad_acc_steps

    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Training Codec ━━━{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Images:{C.RESET}         {C.BOLD}{codec.train_images}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Steps:{C.RESET}          {C.BOLD}{num_steps}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Batch size:{C.RESET}     {batch_size}")
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {minibatch_size}")
    print(f"{C.BRIGHT_CYAN}  Grad acc steps:{C.RESET} {num_grad_acc_steps}")
    print(f"{C.BRIGHT_CYAN}  Learning rate:{C.RESET}  {max_lr}")
    print(f"{C.BRIGHT_CYAN}  LR schedule:{C.RESET}    {codec.lr_schedule}" + (f" (warmup: {warmup_steps} steps / {warmup_images} images)" if warmup_steps > 0 else ""))
    if codec.gradient_clip > 0:
        print(f"{C.BRIGHT_CYAN}  Gradient clip:{C.RESET}  {codec.gradient_clip}")
    # if ema_tracker is not None:
    #     print(f"{C.BRIGHT_CYAN}  EMA decay:{C.RESET}      {velocity.ema}")
    print(f"{C.BRIGHT_CYAN}  Losses:{C.RESET}         {', '.join(l.id for l in codec.loss_fns)}") # type: ignore
    if codec.augmentations:
        print(f"{C.BRIGHT_CYAN}  Augmentations:{C.RESET}  {', '.join(codec.augmentations)}")
    print()

    # Generate augmentation grid at start of training for visual verification
    if codec.augmentations:
        sample_codec_augmentations(rf_model, dataset, run_dir, wandb_run=wandb_run)

    save_minutes = 120
    last_save_time = datetime.datetime.now()

    sample_minutes = 1
    max_sample_minutes = 8
    last_sample_time = datetime.datetime.now()

    def save(step: int):
        save_module(codec, os.path.join(run_dir, f"codec_{run_id}_{step:06d}.ckpt"))
        nonlocal last_save_time
        last_save_time = datetime.datetime.now()

    latest_loss = 0.0
    loss_tracker = AccuracyTracker()  # Reuse for loss smoothing

    progress = tqdm(range(0, last_step), initial=0, total=last_step)
    for step in progress:
        # Update learning rate
        lr = get_lr(step, warmup_steps, num_steps, max_lr, codec.lr_schedule)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        loss_item = 0.0
        loss_details: dict = {}

        for grad_step in range(num_grad_acc_steps):
            try:
                inputs = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs = next(data_iter)

            # Train on real images only (skip random noise domain)
            x1 = inputs["domain_1"].to(device)
            x1 = codec.augment(x1)
            if dataset.use_random_noise_domain0:
                x = x1  # Don't train codec on random noise
            else:
                x0 = inputs["domain_0"].to(device)
                x0 = codec.augment(x0)
                x = torch.cat([x0, x1], dim=0)  # (2B, 3, H, W)

            output = codec.compute_loss(x)
            loss = output["loss"]
            grad_loss: torch.Tensor = loss * grad_scale
            grad_loss.backward()

            loss_item += grad_loss.item()
            # Accumulate per-loss details
            for k, v in output.items():
                if k not in ("loss", "x_hat"):
                    loss_details[k] = loss_details.get(k, 0.0) + v * grad_scale

        # Gradient clipping (after all accumulation, before optimizer step)
        if codec.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=codec.gradient_clip)

        optimizer.step()

        latest_loss = loss_item
        loss_tracker.update(loss_item)

        postfix = {"loss": f"{loss_item:.4f}"}
        for k, v in loss_details.items():
            postfix[k] = f"{v:.4f}"
        postfix["lr"] = f"{lr:.2e}"
        progress.set_postfix(postfix)

        if wandb_run is not None:
            wlog: dict = {"codec_loss": loss_item, "codec_lr": lr, "codec_step": step}
            for k, v in loss_details.items():
                wlog[f"codec_{k}"] = v
            wandb_run.log(wlog)

        if (step == 0) or (datetime.datetime.now() - last_sample_time).total_seconds() >= sample_minutes * 60:
            sample_codec_reconstruction(rf_model, dataset, run_dir, label=f"step_{step:06d}", wandb_run=wandb_run)
            rf_model.train_codec()
            sample_minutes = min(sample_minutes * 2, max_sample_minutes)
            last_sample_time = datetime.datetime.now()

        if (step == 0) or (datetime.datetime.now() - last_save_time).total_seconds() >= save_minutes * 60:
            save(step)

    save(last_step)
    write_codec_state(run_dir, trained=True, loss=latest_loss)
    print(f"\n{C.BOLD}{C.BRIGHT_GREEN}✓ Codec training complete{C.RESET} (loss={C.CYAN}{latest_loss:.6f}{C.RESET})\n")

    # Generate final reconstruction grid for visual verification
    sample_codec_reconstruction(rf_model, dataset, run_dir, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    return latest_loss


@torch.no_grad()
def sample_codec_reconstruction(
    rf_model: RFPix2pixModel,
    dataset: RFPix2pixDataset,
    run_dir: str,
    label: str = "final",
    num_samples: int = 8,
    wandb_run: Optional[wandb.wandb_run.Run] = None,
):
    """Generate a grid showing [original | reconstructed] pairs for codec quality check."""
    rf_model.eval()
    rows = []
    z_shape = None
    def z2img(z: torch.Tensor) -> torch.Tensor:
        nonlocal z_shape
        if z_shape is None:
            z_shape = z.shape
        # Visualize first 3 latent channels, scaled to [-1, 1]
        z_vis = z[:, :3, :, :]  # (1, 3, H', W')
        z_vis = (z_vis / 4.0).clamp(-1, 1)  # scale assuming ~N(0,1) latents
        z_vis = torch.nn.functional.interpolate(z_vis, size=x.shape[2:], mode="nearest")  # upscale
        z_vis = z_vis.clamp(-1, 1)
        return z_vis
    for i in range(num_samples):
        row_images = []
        for domain_index in range(2):
            if domain_index == 0 and dataset.use_random_noise_domain0:
                continue  # Skip random noise domain for reconstruction visualization
            inputs = dataset[random.randint(0, len(dataset) - 1)]
            x = inputs[f"domain_{domain_index}"].unsqueeze(0).to(device)  # (1, 3, H, W)
            enc = rf_model.codec.net.encode_params(x)
            z = enc["z"]  # (1, C, H', W')
            x_hat = rf_model.codec.net.decode(z)
            z_vis = z2img(z)
            row_images.extend([x, x_hat, z_vis])
        if z_shape is not None:
            z = rf_model.codec.net.sample_random_latent(z_shape, device)
            x_dec = rf_model.codec.net.decode(z)
            z_vis = z2img(z)
            row_images.extend([x_dec, z_vis])
        row = torch.cat(row_images, dim=3)  # (1, 3, H, W*M)
        rows.append(row)
    image = torch.cat(rows, dim=2)  # (1, 3, H*N, W*M)
    image = (image.squeeze(0).cpu().numpy() + 1.0) * 127.5
    image = image.clip(0, 255).astype("uint8")
    image = Image.fromarray(image.transpose(1, 2, 0))
    path = os.path.join(run_dir, f"codec_{label}.jpg")
    image.save(path, quality=90)
    print(f"Saved codec grid to {path} ({image.width}x{image.height})")
    if wandb_run is not None:
        wandb_run.log({"codec_reconstruction": wandb.Image(path)})


@torch.no_grad()
def sample_codec_augmentations(
    rf_model: RFPix2pixModel,
    dataset: RFPix2pixDataset,
    run_dir: str,
    num_images: int = 4,
    num_augmentations: int = 6,
    wandb_run: Optional[wandb.wandb_run.Run] = None,
):
    """Generate a grid showing [original | aug1 | aug2 | ...] for each image to visualize augmentations."""
    rf_model.eval()
    codec = rf_model.codec
    rows = []
    for i in range(num_images):
        inputs = dataset[random.randint(0, len(dataset) - 1)]
        x = inputs["domain_1"].unsqueeze(0).to(device)  # (1, 3, H, W)
        cols = [x]
        for _ in range(num_augmentations):
            cols.append(codec.augment(x))
        row = torch.cat(cols, dim=3)  # (1, 3, H, (1+N)*W)
        rows.append(row)
    image = torch.cat(rows, dim=2)  # (1, 3, H*M, (1+N)*W)
    image = (image.squeeze(0).cpu().numpy() + 1.0) * 127.5
    image = image.clip(0, 255).astype("uint8")
    image = Image.fromarray(image.transpose(1, 2, 0))
    path = os.path.join(run_dir, "codec_augmentations.jpg")
    image.save(path, quality=90)
    print(f"Saved codec augmentation grid to {path} ({image.width}x{image.height})")
    if wandb_run is not None:
        wandb_run.log({"codec_augmentations": wandb.Image(path)})
