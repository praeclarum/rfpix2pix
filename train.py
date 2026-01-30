from typing import Optional
import argparse
import os
import sys
import json
import datetime
import torch

from fnn import object_from_config, load_module, device
from data import RFPix2pixDataset
from model import RFPix2pixModel

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
        accuracy: Accuracy as a float (0.0 to 1.0) or percentage (0-100), or None to keep existing
        backbone_warmed_up: Whether backbone warmup is complete, or None to keep existing
    """
    # Read existing state
    state = read_saliency_state(run_dir)
    
    # Update provided fields
    if accuracy is not None:
        # Convert to integer percentage if given as float
        if accuracy <= 1.0:
            accuracy_pct = int(round(accuracy * 100))
        else:
            accuracy_pct = int(round(accuracy))
        state["accuracy"] = accuracy_pct
        print(f"Saved saliency accuracy: {accuracy_pct}%")
    
    if backbone_warmed_up is not None:
        state["backbone_warmed_up"] = backbone_warmed_up
        print(f"Saved backbone_warmed_up: {backbone_warmed_up}")
    
    # Write updated state
    state_path = os.path.join(run_dir, SALIENCY_STATE_FILE)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def should_train_saliency(run_dir: str, threshold: int) -> bool:
    """
    Check if saliency training is needed.
    
    Args:
        run_dir: Run directory path
        threshold: Required accuracy percentage (0-100)
        
    Returns:
        True if saliency training is needed, False otherwise.
    """
    state = read_saliency_state(run_dir)
    accuracy = state["accuracy"]
    if accuracy is None:
        print(f"No saliency checkpoint found, training needed.")
        return True
    if accuracy < threshold:
        print(f"Saliency accuracy {accuracy}% < {threshold}% threshold, training needed.")
        return True
    print(f"Saliency accuracy {accuracy}% >= {threshold}% threshold, skipping saliency training.")
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


def train_saliency(model, dataset, run_dir: str, dev: bool) -> float:
    """
    Train the saliency network until it reaches acceptable accuracy.
    
    Returns:
        Final accuracy as a float (0.0 to 1.0)
    """
    raise NotImplementedError("Training function is not yet implemented.")


def train_velocity(model, dataset, run_dir: str, step_start: int, dev: bool):
    raise NotImplementedError("Training function is not yet implemented.")


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
        "--run-id",
        type=str,
        default=None,
        help="Run ID for output directory (auto-generated if not provided)"
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
    print(f"Loading config from {args.config}")
    config = json.load(open(args.config, "r"))
    
    # Generate run ID
    date_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"run_{date_id}"
        if args.dev:
            run_id += "_dev"
    
    # Load or create model
    step_start = 0
    if args.checkpoint:
        # Extract step from checkpoint filename (e.g., model_1000.ckpt -> 1000)
        step_start = int(args.checkpoint.split("_")[-1].split(".")[0])
        print(f"Loading model from {args.checkpoint}")
        model: RFPix2pixModel = load_module(args.checkpoint).to(device)  # type: ignore
        config = model.__config
    else:
        print("Creating new model")
        model: RFPix2pixModel = object_from_config(config).to(device)
    
    model.compile()
    
    # Create dataset
    dataset = RFPix2pixDataset(
        domain_0_paths=args.domain0,
        domain_1_paths=args.domain1,
        max_size=model.max_size,
        num_downsamples=model.velocity_net.num_downsamples,
    )
    print(f"Dataset: {len(dataset.domain_0_image_paths)} domain 0 images, {len(dataset.domain_1_image_paths)} domain 1 images")
    
    # Create run directory
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save config to run directory
    config_save_path = os.path.join(run_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Phase 1: Train saliency network if needed
    if should_train_saliency(run_dir, model.saliency_accuracy_threshold):
        final_accuracy = train_saliency(model, dataset, run_dir, dev=args.dev)
        write_saliency_state(run_dir, accuracy=final_accuracy)
    
    # Phase 2: Train velocity network
    train_velocity(model, dataset, run_dir, step_start=step_start, dev=args.dev)
