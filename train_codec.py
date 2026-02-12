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

from fnn import save_module, device
from data import RFPix2pixDataset
from model import RFPix2pixModel
from utils import Colors as C, AccuracyTracker

CODEC_STATE_FILE = "codec_state.json"


def read_codec_state(run_dir: str) -> dict:
    """Read the codec training state from the state file."""
    state_path = os.path.join(run_dir, CODEC_STATE_FILE)
    default_state = {
        "trained": False,
        "loss": None,
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


def write_codec_state(run_dir: str, trained: bool | None = None, loss: float | None = None, backbone_warmed_up: bool | None = None):
    """Update the codec training state file."""
    state = read_codec_state(run_dir)
    if trained is not None:
        state["trained"] = trained
    if loss is not None:
        state["loss"] = loss
        print(f"{C.DIM}Saved codec loss: {C.CYAN}{loss:.6f}{C.RESET}")
    if backbone_warmed_up is not None:
        state["backbone_warmed_up"] = backbone_warmed_up
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

    codec = rf_model.codec
    lr = codec.learning_rate
    batch_size = codec.train_batch_size if codec.train_batch_size is not None else 48
    minibatch_size = codec.train_minibatch_size if codec.train_minibatch_size is not None else batch_size

    num_steps = codec.train_images // batch_size
    last_step = num_steps

    # Check if backbone warmup was already completed (resuming)
    codec_state = read_codec_state(run_dir)
    backbone_already_warmed = codec_state["backbone_warmed_up"]

    def create_optimizer(backbone_frozen: bool) -> torch.optim.AdamW:
        param_groups = codec.get_optimizer_param_groups(lr, backbone_frozen)
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-4)

    has_backbone = codec.has_backbone()
    if has_backbone and not backbone_already_warmed:
        codec.freeze_backbone()
        backbone_frozen = True
    elif has_backbone and backbone_already_warmed:
        codec.unfreeze_backbone()
        backbone_frozen = False
    else:
        backbone_frozen = False

    optimizer = create_optimizer(backbone_frozen)

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
    print(f"{C.BRIGHT_CYAN}  Steps:{C.RESET}          {C.BOLD}{num_steps}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Batch size:{C.RESET}     {batch_size}")
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {minibatch_size}")
    print(f"{C.BRIGHT_CYAN}  Grad acc steps:{C.RESET} {num_grad_acc_steps}")
    print(f"{C.BRIGHT_CYAN}  Learning rate:{C.RESET}  {lr}")
    print(f"{C.BRIGHT_CYAN}  Losses:{C.RESET}         {', '.join(l.id for l in codec.loss_fns)}") # type: ignore
    if has_backbone:
        print(f"{C.BRIGHT_CYAN}  Backbone frozen:{C.RESET} {C.YELLOW if backbone_frozen else C.GREEN}{backbone_frozen}{C.RESET}")
    print()

    save_minutes = 30
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
            if dataset.use_random_noise_domain0:
                x = x1  # Don't train codec on random noise
            else:
                x0 = inputs["domain_0"].to(device)
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

        optimizer.step()
        gc.collect()

        latest_loss = loss_item
        loss_tracker.update(loss_item)

        postfix = {"loss": f"{loss_item:.4f}"}
        for k, v in loss_details.items():
            postfix[k] = f"{v:.4f}"
        progress.set_postfix(postfix)

        if wandb_run is not None:
            wlog: dict = {"codec_loss": loss_item, "codec_lr": lr, "codec_step": step}
            for k, v in loss_details.items():
                wlog[f"codec_{k}"] = v
            wandb_run.log(wlog)

        if (datetime.datetime.now() - last_sample_time).total_seconds() >= sample_minutes * 60:
            sample_codec_reconstruction(rf_model, dataset, run_dir, label=f"step_{step:06d}", wandb_run=wandb_run)
            rf_model.train_codec()
            sample_minutes = min(sample_minutes * 2, max_sample_minutes)
            last_sample_time = datetime.datetime.now()

        if (datetime.datetime.now() - last_save_time).total_seconds() >= save_minutes * 60:
            save(step)

        # Backbone warmup transition (if applicable)
        if has_backbone and backbone_frozen and codec.warmup_threshold is not None:
            if loss_tracker.is_stable and loss_tracker.smoothed < codec.warmup_threshold:
                print(f"\n{C.BOLD}{C.GREEN}✓ Codec backbone warmup complete{C.RESET} at step {C.CYAN}{step}{C.RESET}")
                codec.unfreeze_backbone()
                optimizer = create_optimizer(backbone_frozen=False)
                backbone_frozen = False
                loss_tracker.reset()
                save(step)
                write_codec_state(run_dir, backbone_warmed_up=True)
                if wandb_run is not None:
                    wandb_run.log({"codec_backbone_unfrozen": 1, "codec_step": step})

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
    for i in range(num_samples):
        inputs = dataset[random.randint(0, len(dataset) - 1)]
        x = inputs["domain_1"].unsqueeze(0).to(device)  # (1, 3, H, W)
        x_hat = rf_model.codec(x)  # encode -> decode
        row = torch.cat([x, x_hat], dim=3)  # (1, 3, H, 2W)
        rows.append(row)
    image = torch.cat(rows, dim=2)  # (1, 3, H*N, 2W)
    image = (image.squeeze(0).cpu().numpy() + 1.0) * 127.5
    image = image.clip(0, 255).astype("uint8")
    image = Image.fromarray(image.transpose(1, 2, 0))
    path = os.path.join(run_dir, f"codec_{label}.jpg")
    image.save(path, quality=90)
    print(f"Saved codec grid to {path} ({image.width}x{image.height})")
    if wandb_run is not None:
        wandb_run.log({"codec_reconstruction": wandb.Image(path)})
