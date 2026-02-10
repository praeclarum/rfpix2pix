import gc
import random
from contextlib import nullcontext
from typing import Optional
import argparse
import os
import json
import datetime
import shutil

import numpy as np
import torch
from tqdm import tqdm
import wandb
import wandb.wandb_run
from PIL import Image

from fnn import object_from_config, load_module, device, save_module
from data import RFPix2pixDataset, StructurePairing, load_media_item, prepare_structure_pairing
from model import RFPix2pixModel
from utils import Colors as C, AccuracyTracker

SALIENCY_STATE_FILE = "saliency_state.json"


def read_saliency_state(run_dir: str) -> dict:
    """
    Read the saliency training state from the state file.
    
    Returns:
        dict with keys:
            - accuracy: int percentage (0-100) or None if not yet trained
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
            # Ensure all expected keys exist
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
    
    Args:
        run_dir: Run directory path
        accuracy: Accuracy as a float (0.0 to 1.0), or None to keep existing
        backbone_warmed_up: Whether backbone warmup is complete, or None to keep existing
    """
    # Read existing state
    state = read_saliency_state(run_dir)
    
    # Update provided fields
    if accuracy is not None:
        # Convert to integer percentage if given as float
        state["accuracy"] = accuracy
        print(f"{C.DIM}Saved saliency accuracy: {C.CYAN}{accuracy*100:.2f}%{C.RESET}")
    
    if backbone_warmed_up is not None:
        state["backbone_warmed_up"] = backbone_warmed_up
        print(f"{C.DIM}Saved backbone_warmed_up: {C.CYAN}{backbone_warmed_up}{C.RESET}")
    
    # Write updated state
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
    """
    Check if backbone warmup phase is complete.
    
    Args:
        run_dir: Run directory path
        
    Returns:
        True if backbone warmup is complete, False otherwise.
    """
    state = read_saliency_state(run_dir)
    return state["backbone_warmed_up"]


def train_saliency(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, dev: bool) -> float:
    """
    Train the saliency network until it reaches acceptable accuracy.
    
    Returns:
        Final accuracy as a float (0.0 to 1.0)
    """
    rf_model.train_saliency()
    rf_model.to(device)

    saliency_net = rf_model.saliency_net

    num_steps = rf_model.train_images // rf_model.train_batch_size
    last_step = num_steps
    
    lr = rf_model.saliency_learning_rate
    
    # Check if backbone warmup was already completed (resuming training)
    backbone_already_warmed = is_backbone_warmed_up(run_dir)
    
    def create_optimizer(backbone_frozen: bool) -> torch.optim.AdamW:
        """Create optimizer with appropriate parameter groups."""
        param_groups = saliency_net.get_optimizer_param_groups(lr, backbone_frozen)
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-4)
    
    if backbone_already_warmed:
        # Resume with unfrozen backbone
        saliency_net.unfreeze_backbone()
        backbone_frozen = False
    else:
        # Start with frozen backbone for warmup
        saliency_net.freeze_backbone()
        backbone_frozen = True
    
    optimizer = create_optimizer(backbone_frozen)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=rf_model.train_minibatch_size, shuffle=True, num_workers=8)
    data_iter = iter(dataloader)

    # Extract run_id from run_dir for logging
    run_id = os.path.basename(run_dir)

    def save(step: int):
        save_module(rf_model, os.path.join(run_dir, f"rfpix2pix_{run_id}_{step:06d}.ckpt"))

    num_grad_acc_steps = max(1, rf_model.train_batch_size // rf_model.train_minibatch_size)
    grad_scale = 1.0 / num_grad_acc_steps
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Training Saliency Network ━━━{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Steps:{C.RESET}          {C.BOLD}{num_steps}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {rf_model.train_minibatch_size}")
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
            input_domain_0 = rf_model.saliency_augment(input_domain_0)
            input_domain_1 = rf_model.saliency_augment(input_domain_1)
            
            output = rf_model.compute_saliency_loss(input_domain_0, input_domain_1)
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
        if backbone_frozen and accuracy_tracker.above_threshold(rf_model.saliency_warmup_threshold):
            print(f"\n{C.BOLD}{C.GREEN}✓ Backbone warmup complete{C.RESET} at step {C.CYAN}{step}{C.RESET}, smoothed accuracy {C.BRIGHT_GREEN}{smoothed_accuracy*100:.2f}%{C.RESET}")
            saliency_net.unfreeze_backbone()
            # Create new optimizer with differential learning rates
            optimizer = create_optimizer(backbone_frozen=False)
            backbone_frozen = False
            # Reset tracker when entering new phase
            accuracy_tracker.reset()
            save(step)
            write_saliency_state(run_dir, accuracy=smoothed_accuracy, backbone_warmed_up=True)
        elif not backbone_frozen and accuracy_tracker.above_threshold(rf_model.saliency_accuracy_threshold):
            print(f"\n{C.BOLD}{C.BRIGHT_GREEN}✓ Saliency training complete{C.RESET} at step {C.CYAN}{step}{C.RESET}, smoothed accuracy {C.BRIGHT_GREEN}{smoothed_accuracy*100:.2f}%{C.RESET}\n")
            save(step)
            return accuracy_tracker.smoothed
        
    # Training ended without reaching threshold - return smoothed accuracy if available
    save(last_step)
    return accuracy_tracker.smoothed if accuracy_tracker.is_stable else accuracy_item

@torch.no_grad()
def sample(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, name: str):
    # Generate sample grid
    rows = []
    for i in range(rf_model.sample_batch_size):
        inputs = dataset[random.randint(0, len(dataset) - 1)]
        input_image = inputs["domain_0"].unsqueeze(0).to(device)  # (1, 3, H, W)
        output_image = rf_model.generate(input_image)
        row = torch.cat([input_image, output_image], dim=3)  # (1, 3, H, 2W)
        rows.append(row)
    image = torch.cat(rows, dim=2)  # (1, 3, H*B, 2W)
    image = (image.squeeze(0).cpu().numpy() + 1.0) * 127.5
    image = image.clip(0, 255).astype("uint8")
    image = Image.fromarray(image.transpose(1, 2, 0))
    # Save the sample grid image
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
    
    Args:
        dataset: Dataset with media items for both domains
        pairing: StructurePairing object with precomputed similarities
        run_dir: Directory to save the proof sheet
        num_samples: Number of domain 0 images to sample (rows in grid)
    """
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Structure Pairing Proof Sheet ━━━{C.RESET}")
    
    n_domain_0 = len(dataset.domain_0_media_items)
    k = pairing.top_k_indices.shape[1]  # Number of candidates per image
    
    # Sample random domain 0 indices
    if num_samples >= n_domain_0:
        domain_0_indices = list(range(n_domain_0))
    else:
        domain_0_indices = random.sample(range(n_domain_0), num_samples)
    
    rows = []
    for d0_idx in tqdm(domain_0_indices, desc="Building proof sheet"):
        # Load domain 0 image
        d0_item = dataset.domain_0_media_items[d0_idx]
        d0_image = load_media_item(
            d0_item,
            dataset.max_image_size,
        )  # (3, H, W)
        
        # Get top-K domain 1 candidates (already sorted by similarity)
        d1_indices = pairing.top_k_indices[d0_idx]  # (K,)
        
        # Build row: [domain_0, match_1, match_2, ..., match_K]
        row_images = [d0_image]
        for d1_idx in d1_indices:
            d1_item = dataset.domain_1_media_items[d1_idx]
            d1_image = load_media_item(
                d1_item,
                dataset.max_image_size,
            )  # (3, H, W)
            row_images.append(d1_image)
        
        # Concatenate horizontally: (3, H, W*(K+1))
        row = torch.cat([img.unsqueeze(0) for img in row_images], dim=3)  # (1, 3, H, W*(K+1))
        rows.append(row)
    
    # Stack all rows vertically: (1, 3, H*num_samples, W*(K+1))
    grid = torch.cat(rows, dim=2)
    
    # Convert from [-1, 1] to [0, 255]
    grid = (grid.squeeze(0).cpu().numpy() + 1.0) * 127.5
    grid = grid.clip(0, 255).astype("uint8")
    grid = Image.fromarray(grid.transpose(1, 2, 0))
    
    # Save the proof sheet
    path = os.path.join(run_dir, "pairings.jpg")
    grid.save(path, quality=90)
    print(f"{C.GREEN}✓ Saved structure pairing proof sheet to {C.BOLD}{path}{C.RESET}")
    print(f"  {C.DIM}Grid: {len(domain_0_indices)} rows × {k + 1} columns (source + {k} matches){C.RESET}\n")


def train_velocity(rf_model: RFPix2pixModel, dataset: RFPix2pixDataset, run_dir: str, step_start: int, dev: bool):
    """
    Train the velocity network using saliency-weighted flow matching loss.
    
    The saliency network is frozen and used to compute gradients that weight
    the velocity loss. This focuses learning on coordinates that matter for
    style transfer.
    
    Args:
        rf_model: The RFPix2pixModel with trained saliency network
        dataset: Dataset providing paired domain images
        run_dir: Directory for saving checkpoints
        step_start: Step to resume from (0 for fresh start)
        dev: If True, skip wandb logging
    """
    # Ensure saliency is frozen, velocity is trainable
    rf_model.eval_saliency()
    rf_model.velocity_net.train()
    rf_model.to(device)

    velocity_net = rf_model.velocity_net

    num_steps = rf_model.train_images // rf_model.train_batch_size
    last_step = step_start + num_steps
    
    lr = rf_model.learning_rate
    
    optimizer = torch.optim.AdamW(
        velocity_net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=rf_model.train_minibatch_size,
        shuffle=True,
        num_workers=8
    )
    data_iter = iter(dataloader)

    # Extract run_id from run_dir for logging
    run_id = os.path.basename(run_dir)

    if dev:
        wandb_run: Optional[wandb.wandb_run.Run] = None
    else:
        wandb_run: Optional[wandb.wandb_run.Run] = wandb.init(
            project="rfpix2pix_gen" if rf_model.is_generative else "rfpix2pix",
            save_code=True,
            id=run_id,
            config=rf_model.__config,  # pyright: ignore[reportArgumentType]
        )
        wandb_run.watch(velocity_net)

    def save(step: int):
        save_module(rf_model, os.path.join(run_dir, f"rfpix2pix_{run_id}_{step:06d}.ckpt"))

    num_grad_acc_steps = max(1, rf_model.train_batch_size // rf_model.train_minibatch_size)
    grad_scale = 1.0 / num_grad_acc_steps
    
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Training Velocity Network ━━━{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Steps:{C.RESET}          {C.BOLD}{num_steps}{C.RESET}")
    print(f"{C.BRIGHT_CYAN}  Starting step:{C.RESET}  {step_start}")
    print(f"{C.BRIGHT_CYAN}  Batch size:{C.RESET}     {rf_model.train_batch_size}")
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {rf_model.train_minibatch_size}")
    print(f"{C.BRIGHT_CYAN}  Grad acc steps:{C.RESET} {num_grad_acc_steps}")
    print(f"{C.BRIGHT_CYAN}  Learning rate:{C.RESET}  {lr}\n")

    sample_minutes = 1
    max_sample_minutes = 8
    last_sample_time = datetime.datetime.now()

    save_minutes = 30
    last_save_time = datetime.datetime.now()

    progress = tqdm(range(step_start, last_step), initial=step_start, total=last_step - step_start)
    for step in progress:
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
            autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if rf_model.bf16 else nullcontext() # pyright: ignore[reportPrivateImportUsage]
            with autocast_ctx:
                output = rf_model.compute_loss(input_domain_0, input_domain_1)
                loss = output['loss']
            grad_loss: torch.Tensor = loss * grad_scale
            grad_loss.backward()

            loss_item += grad_loss.item()

        optimizer.step()
        gc.collect()
        
        progress.set_postfix({
            "loss": f"{loss_item:.4f}",
        })
        
        if wandb_run is not None:
            wlog = {
                "vel_loss": loss_item,
                "lr": lr,
                "step": step,
            }
            wandb_run.log(wlog)

        if (datetime.datetime.now() - last_sample_time).total_seconds() >= sample_minutes * 60:
            sample(rf_model, dataset, run_dir, f"step_{step:06d}")
            sample_minutes = min(sample_minutes * 2, max_sample_minutes)
            last_sample_time = datetime.datetime.now()
        
        if (datetime.datetime.now() - last_save_time).total_seconds() >= save_minutes * 60:
            save(step)
            last_save_time = datetime.datetime.now()

    # Final save
    save(last_step)
    print(f"\n{C.BOLD}{C.BRIGHT_GREEN}✓ Velocity training complete{C.RESET} at step {C.CYAN}{last_step}{C.RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF Pix2Pix model for unpaired image-to-image translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python train.py --config configs/small.json --domain0 data/horses --domain1 data/zebras

  # Resume from checkpoint
  python train.py --config configs/small.json --domain0 data/horses --domain1 data/zebras --checkpoint runs/run_xxx/model_1000.ckpt

  # Dev mode (smaller batches, more logging)
  python train.py --config configs/small.json --domain0 data/horses --domain1 data/zebras --dev
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to model config JSON file"
    )
    parser.add_argument(
        "--domain0", "-d0",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to domain 0 image directories"
    )
    parser.add_argument(
        "--domain1", "-d1",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to domain 1 image directories"
    )
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode (smaller batches, more frequent logging)"
    )
    parser.add_argument(
        "--saliency-only",
        action="store_true",
        help="Only train the saliency network (Phase 1), skip velocity training"
    )
    
    return parser.parse_args()


#
# APPLICATION
#
if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    print(f"{C.BLUE}▶ Loading config from {C.BOLD}{args.config}{C.RESET}")
    config = json.load(open(args.config, "r"))
    
    # Generate run ID
    date_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    run_id = f"run_{date_id}"
    if args.dev:
        run_id += "_dev"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"{C.BLUE}▶ Run directory:{C.RESET} {C.BOLD}{run_dir}{C.RESET}\n")
    
    # Load or create model
    step_start = 0
    if args.checkpoint:
        # Extract step from checkpoint filename (e.g., model_1000.ckpt -> 1000)
        step_start = int(args.checkpoint.split("_")[-1].split(".")[0])
        print(f"{C.BLUE}▶ Loading model from {C.BOLD}{args.checkpoint}{C.RESET}")
        model: RFPix2pixModel = load_module(args.checkpoint).to(device)  # type: ignore
        config = model.__config
        prev_run_dir = os.path.dirname(args.checkpoint)
        prev_saliency_state_path = os.path.join(prev_run_dir, SALIENCY_STATE_FILE)
        if os.path.exists(prev_saliency_state_path):
            shutil.copy(
                os.path.join(prev_run_dir, SALIENCY_STATE_FILE),
                os.path.join(run_dir, SALIENCY_STATE_FILE)
            )
        try:
            data_paths = json.load(open(os.path.join(prev_run_dir, "data.json"), "r"))
            args.domain0 = data_paths["domain0_paths"]
            args.domain1 = data_paths["domain1_paths"]
        except Exception as e:
            print(f"{C.RED}▶ Warning: Failed to restore data paths from previous run: {e}{C.RESET}")
        print(f"{C.BLUE}▶ Resuming from step {C.CYAN}{step_start}{C.RESET}\n")
    else:
        print(f"{C.BLUE}▶ Creating new model{C.RESET}")
        model: RFPix2pixModel = object_from_config(config).to(device)
    
    model.compile()
    
    # Create dataset
    dataset = RFPix2pixDataset(
        domain_0_paths=args.domain0,
        domain_1_paths=args.domain1,
        max_size=model.max_size,
        num_downsamples=model.velocity_net.num_downsamples,
    )
    if dataset.use_random_noise_domain0:
        print(
            f"{C.BLUE}▶ Dataset:{C.RESET} domain 0 = {C.CYAN}random noise{C.RESET}, "
            f"{C.CYAN}{len(dataset.domain_1_media_items)}{C.RESET} domain 1 media frames"
        )
    else:
        print(
            f"{C.BLUE}▶ Dataset:{C.RESET} {C.CYAN}{len(dataset.domain_0_media_items)}{C.RESET} "
            f"domain 0 media frames, {C.CYAN}{len(dataset.domain_1_media_items)}{C.RESET} domain 1 media frames"
        )
    
    # Save config to run directory
    config_save_path = os.path.join(run_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"{C.BLUE}▶ Saved config to {C.BOLD}{config_save_path}{C.RESET}\n")

    # Save data paths to run directory
    data_save_path = os.path.join(run_dir, "data.json")
    with open(data_save_path, "w") as f:
        json.dump({
            "domain0_paths": args.domain0,
            "domain1_paths": args.domain1,
        }, f, indent=2)
    print(f"{C.BLUE}▶ Saved data to {C.BOLD}{data_save_path}{C.RESET}\n")

    # Sample to make sure everything is working
    sample(model, dataset, run_dir, f"init_{step_start}")
    
    # Phase 0: Prepare structure pairing if enabled
    structure_pairing: Optional[StructurePairing] = None
    if model.structure_pairing:
        structure_pairing = prepare_structure_pairing(
            dataset,
            structure_candidates=model.structure_candidates,
            device=device,
        )
        # Generate proof sheet for debugging structure pairings
        sample_structure_pairings(dataset, structure_pairing, run_dir)

    # Set generative mode flag on model
    model.is_generative = dataset.use_random_noise_domain0
    
    # Phase 1: Train saliency network if needed (skip in generative mode)
    if model.is_generative:
        print(f"{C.CYAN}ℹ Generative mode: skipping saliency network training (simple MSE loss).{C.RESET}")
    elif should_train_saliency(run_dir, model.saliency_accuracy_threshold):
        final_accuracy = train_saliency(model, dataset, run_dir, dev=args.dev)
        write_saliency_state(run_dir, accuracy=final_accuracy)
    
    if args.saliency_only:
        print(f"{C.GREEN}✓ Saliency-only mode: skipping velocity training.{C.RESET}")
    else:
        if structure_pairing is not None:
            # Recreate dataset with structure pairing
            dataset = RFPix2pixDataset(
                domain_0_paths=args.domain0,
                domain_1_paths=args.domain1,
                max_size=model.max_size,
                num_downsamples=model.velocity_net.num_downsamples,
                structure_pairing=structure_pairing,
            )
        
        # Phase 2: Train velocity network
        train_velocity(model, dataset, run_dir, step_start=step_start, dev=args.dev)
