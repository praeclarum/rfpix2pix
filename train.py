"""
RF Pix2Pix training orchestrator.

Coordinates three training phases:
    Phase 0: Codec training (encode/decode between pixel and latent space)
    Phase 1: Saliency training (domain classifier for loss weighting)
    Phase 2: Velocity training (flow matching in latent space)

Each phase has its own training script (train_codec, train_saliency, train_velocity)
and saves per-module checkpoints independently.
"""
import argparse
import os
import json
import datetime
import shutil

from fnn import object_from_config, load_module, device, load_config
from data import RFPix2pixDataset, StructurePairing, prepare_structure_pairing
from model import RFPix2pixModel
from train_codec import should_train_codec, train_codec, CODEC_STATE_FILE
from train_saliency import should_train_saliency, train_saliency, write_saliency_state, SALIENCY_STATE_FILE
from train_velocity import train_velocity, sample, sample_structure_pairings
from utils import Colors as C

from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF Pix2Pix model for unpaired image-to-image translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python3 train.py --config configs/medium_vae_new.json --domain0 data/horses --domain1 data/zebras

  # Resume with pretrained codec
  python3 train.py --config configs/medium_vae_new.json --domain0 data/horses --domain1 data/zebras \\
      --codec-ckpt runs/run_codec/codec_final.ckpt

  # Dev mode (skip wandb)
  python3 train.py --config configs/medium_vae_new.json --domain0 data/horses --domain1 data/zebras --dev
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
        default=None,
        help="Path(s) to domain 0 image directories (overrides config)"
    )
    parser.add_argument(
        "--domain1", "-d1",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to domain 1 image directories (overrides config)"
    )
    parser.add_argument(
        "--codec-ckpt",
        type=str,
        default=None,
        help="Path to pretrained codec checkpoint (skips codec training)"
    )
    parser.add_argument(
        "--saliency-ckpt",
        type=str,
        default=None,
        help="Path to pretrained saliency checkpoint (skips saliency training)"
    )
    parser.add_argument(
        "--velocity-ckpt",
        type=str,
        default=None,
        help="Path to pretrained velocity checkpoint (resumes velocity training)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode (skip wandb logging)"
    )
    parser.add_argument(
        "--saliency-only",
        action="store_true",
        help="Only train the saliency network (Phase 1), skip velocity training"
    )
    parser.add_argument(
        "--codec-only",
        action="store_true",
        help="Only train the codec (Phase 0), skip saliency and velocity training"
    )
    
    return parser.parse_args()


def _copy_state_file(src_dir: str, dst_dir: str, filename: str):
    """Copy a state file from one run directory to another if it exists."""
    src = os.path.join(src_dir, filename)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(dst_dir, filename))


#
# APPLICATION
#
if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    print(f"{C.BLUE}▶ Loading config from {C.BOLD}{args.config}{C.RESET}")
    config = load_config(args.config)
    
    # Resolve domain paths: CLI overrides config
    domain0_paths = args.domain0 if args.domain0 is not None else config.get("domain0", [])
    domain1_paths = args.domain1 if args.domain1 is not None else config.get("domain1", [])
    
    if not domain0_paths or not domain1_paths:
        print(f"{C.RED}Error: domain0 and domain1 must be specified via CLI (--domain0, --domain1) or in config.{C.RESET}")
        exit(1)
    
    # Generate run ID
    date_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    run_id = f"run_{date_id}"
    if args.dev:
        run_id += "_dev"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"{C.BLUE}▶ Run directory:{C.RESET} {C.BOLD}{run_dir}{C.RESET}\n")
    
    # Create model from config
    print(f"{C.BLUE}▶ Creating model{C.RESET}")
    model: RFPix2pixModel = object_from_config(config).to(device)
    
    # Load pretrained module checkpoints if provided
    step_start = 0
    if args.codec_ckpt:
        print(f"{C.BLUE}▶ Loading codec from {C.BOLD}{args.codec_ckpt}{C.RESET}")
        model.codec = load_module(args.codec_ckpt).to(device)
        # Copy codec state from the checkpoint's run directory
        prev_run_dir = os.path.dirname(args.codec_ckpt)
        _copy_state_file(prev_run_dir, run_dir, CODEC_STATE_FILE)
    
    if args.saliency_ckpt:
        print(f"{C.BLUE}▶ Loading saliency from {C.BOLD}{args.saliency_ckpt}{C.RESET}")
        model.saliency = load_module(args.saliency_ckpt).to(device)
        # Copy saliency state from the checkpoint's run directory
        prev_run_dir = os.path.dirname(args.saliency_ckpt)
        _copy_state_file(prev_run_dir, run_dir, SALIENCY_STATE_FILE)
    
    if args.velocity_ckpt:
        print(f"{C.BLUE}▶ Loading velocity from {C.BOLD}{args.velocity_ckpt}{C.RESET}")
        model.velocity = load_module(args.velocity_ckpt).to(device)
        # Extract step from checkpoint filename (e.g., velocity_run_xxx_001000.ckpt -> 1000)
        step_start = int(args.velocity_ckpt.split("_")[-1].split(".")[0])
        print(f"{C.BLUE}▶ Resuming velocity from step {C.CYAN}{step_start}{C.RESET}")
    
    model.compile()
    
    # Create dataset
    dataset = RFPix2pixDataset(
        domain_0_paths=domain0_paths,
        domain_1_paths=domain1_paths,
        max_size=model.max_size,
        num_downsamples=model.velocity.num_downsamples,
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
            "domain0_paths": domain0_paths,
            "domain1_paths": domain1_paths,
        }, f, indent=2)
    print(f"{C.BLUE}▶ Saved data to {C.BOLD}{data_save_path}{C.RESET}\n")

    # Sample to make sure everything is working
    sample(model, dataset, run_dir, f"init_{step_start}")
    
    # Prepare structure pairing if enabled
    structure_pairing: Optional[StructurePairing] = None
    if model.velocity.structure_pairing:
        structure_pairing = prepare_structure_pairing(
            dataset,
            structure_candidates=model.velocity.structure_candidates,
            device=device,
        )
        sample_structure_pairings(dataset, structure_pairing, run_dir)

    is_generative = dataset.use_random_noise_domain0
    
    # Phase 0: Train codec if needed
    if should_train_codec(model, run_dir):
        train_codec(model, dataset, run_dir, dev=args.dev)
    # Freeze codec for all downstream phases
    model.eval_codec()
    
    if args.codec_only:
        print(f"{C.GREEN}✓ Codec-only mode: skipping saliency and velocity training.{C.RESET}")
    else:
        # Phase 1: Train saliency network if needed (skip in generative mode)
        if is_generative:
            print(f"{C.CYAN}ℹ Generative mode: skipping saliency network training (simple MSE loss).{C.RESET}")
        elif should_train_saliency(run_dir, model.saliency.accuracy_threshold):
            final_accuracy = train_saliency(model, dataset, run_dir, dev=args.dev)
            write_saliency_state(run_dir, accuracy=final_accuracy)
    
        if args.saliency_only:
            print(f"{C.GREEN}✓ Saliency-only mode: skipping velocity training.{C.RESET}")
        else:
            if structure_pairing is not None:
                # Recreate dataset with structure pairing
                dataset = RFPix2pixDataset(
                    domain_0_paths=domain0_paths,
                    domain_1_paths=domain1_paths,
                    max_size=model.max_size,
                    num_downsamples=model.velocity.num_downsamples,
                    structure_pairing=structure_pairing,
                )
            
            # Phase 2: Train velocity network
            train_velocity(model, dataset, run_dir, step_start=step_start, dev=args.dev)
