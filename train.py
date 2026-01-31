import gc
import random
from typing import Optional
import argparse
import os
import json
import datetime
import shutil

import torch
from tqdm import tqdm
import wandb
import wandb.wandb_run
from PIL import Image

from fnn import object_from_config, load_module, device, save_module
from data import RFPix2pixDataset
from model import RFPix2pixModel

from utils import Colors as C
from utils import AccuracyTracker

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
    
    lr = rf_model.learning_rate
    
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

    save_steps = 512
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
            project="rfpix2pix",
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
    print(f"{C.BRIGHT_CYAN}  Minibatch size:{C.RESET} {rf_model.train_minibatch_size}")
    print(f"{C.BRIGHT_CYAN}  Grad acc steps:{C.RESET} {num_grad_acc_steps}")
    print(f"{C.BRIGHT_CYAN}  Learning rate:{C.RESET}  {lr}\n")

    sample_steps = 32
    max_sample_steps = 256
    next_sample_step = step_start + sample_steps

    save_steps = 512
    next_save_step = step_start + save_steps

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

        if step >= next_sample_step:
            sample(rf_model, dataset, run_dir, f"step_{step:06d}")
            sample_steps = min(sample_steps * 2, max_sample_steps)
            next_sample_step = step + sample_steps
        
        if step >= next_save_step:
            save(step)
            next_save_step = step + save_steps

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
        shutil.copy(
            os.path.join(prev_run_dir, SALIENCY_STATE_FILE),
            os.path.join(run_dir, SALIENCY_STATE_FILE)
        )
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
    print(f"{C.BLUE}▶ Dataset:{C.RESET} {C.CYAN}{len(dataset.domain_0_image_paths)}{C.RESET} domain 0 images, {C.CYAN}{len(dataset.domain_1_image_paths)}{C.RESET} domain 1 images")
    
    # Save config to run directory
    config_save_path = os.path.join(run_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    # Sample to make sure everything is working
    sample(model, dataset, run_dir, f"init_{step_start}")
    
    # Phase 1: Train saliency network if needed
    if should_train_saliency(run_dir, model.saliency_accuracy_threshold):
        final_accuracy = train_saliency(model, dataset, run_dir, dev=args.dev)
        write_saliency_state(run_dir, accuracy=final_accuracy)
    
    # Phase 2: Train velocity network
    train_velocity(model, dataset, run_dir, step_start=step_start, dev=args.dev)
