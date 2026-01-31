"""
Split images into domain directories using a trained saliency network.

This script loads a checkpoint, classifies images using the saliency network,
and organizes them into domain-specific output directories with MD5-based
deduplication and an ignore list.
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from fnn import device, load_module
from model import RFPix2pixModel
from utils import Colors as C, compute_file_md5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split images into domain directories using a trained saliency network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split images using a trained model
  python split.py --checkpoint runs/run_xxx/model.ckpt --input /path/to/images --output /path/to/output

  # With custom threshold and batch size
  python split.py -ckpt runs/run_xxx/model.ckpt -i images1 images2 -o output --threshold 0.8 --batch-size 32

  # Split with max image size of 1024px
  python split.py -ckpt runs/run_xxx/model.ckpt -i images -o output --max-size 1024

  # Resize existing output images to max 1024px
  python split.py -ckpt runs/run_xxx/model.ckpt -i . -o output --max-size 1024 --resize-existing
        """
    )
    
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file with trained saliency network"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input image directories"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (will contain domain0, domain1, uncertain, ignored subdirs)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Confidence threshold for classification (default: 0.7). Images below this go to 'uncertain'"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--max-size", "-s",
        type=int,
        default=None,
        help="Maximum dimension for output images. Images larger than this will be resized (default: no resizing)"
    )
    parser.add_argument(
        "--resize-existing",
        action="store_true",
        help="Resize existing images in output directory to meet --max-size constraint and exit"
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="JPEG/WebP quality for resized images (default: 95)"
    )
    
    return parser.parse_args()


class ProgressiveImageScanner:
    """
    Progressively scans directories for images using a queue-based approach.
    
    Instead of scanning all directories upfront, this yields images as they're
    discovered, allowing processing to start immediately while scanning continues.
    Subdirectories are queued and processed after their parent's images are yielded.
    """
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, directories: List[str]):
        """
        Initialize the scanner with a list of starting directories.
        
        Args:
            directories: List of directory paths to scan
        """
        from collections import deque
        self.dir_queue: deque[Path] = deque()
        self.images_found = 0
        self.dirs_scanned = 0
        
        # Seed the queue with initial directories
        for d in directories:
            path = Path(d)
            if path.exists() and path.is_dir():
                self.dir_queue.append(path)
            else:
                print(f"{C.YELLOW}⚠ Directory not found: {d}{C.RESET}")
    
    def __iter__(self):
        """Iterate over image paths progressively."""
        return self._scan()
    
    def _scan(self):
        """Generator that yields image paths while progressively scanning directories."""
        while self.dir_queue:
            current_dir = self.dir_queue.popleft()
            self.dirs_scanned += 1
            
            try:
                # List directory contents once
                entries = list(current_dir.iterdir())
            except PermissionError:
                print(f"{C.YELLOW}⚠ Permission denied: {current_dir}{C.RESET}")
                continue
            except OSError as e:
                print(f"{C.YELLOW}⚠ Error reading {current_dir}: {e}{C.RESET}")
                continue
            
            subdirs = []
            images = []
            
            for entry in entries:
                if entry.is_dir():
                    subdirs.append(entry)
                elif entry.is_file() and entry.suffix.lower() in self.IMAGE_EXTENSIONS:
                    images.append(entry)
            
            # Yield images from this directory first
            for img_path in images:
                self.images_found += 1
                yield str(img_path)
            
            # Then queue subdirectories for later processing
            for subdir in sorted(subdirs):
                self.dir_queue.append(subdir)
    
    @property
    def status(self) -> str:
        """Return a status string for progress display."""
        return f"dirs:{self.dirs_scanned} queued:{len(self.dir_queue)}"


def get_existing_hashes(output_dir: Path) -> Set[str]:
    """Get MD5 hashes of files already in output subdirectories."""
    hashes = set()
    subdirs = ["domain0", "domain1", "uncertain", "ignored"]
    
    for subdir in subdirs:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            for file_path in subdir_path.iterdir():
                if file_path.is_file():
                    # Extract hash from filename (format: {hash}.{ext})
                    hash_part = file_path.stem
                    if len(hash_part) == 32:  # MD5 hash length
                        hashes.add(hash_part.lower())
    
    return hashes


def resize_image_to_max_size(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resize an image so its largest dimension is <= max_size.
    Maintains aspect ratio using LANCZOS resampling.
    Returns the original image if already within limits.
    """
    w, h = image.size
    max_dim = max(w, h)
    
    if max_dim <= max_size:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return image.resize((new_w, new_h), Image.LANCZOS)


def save_image_with_format(image: Image.Image, dest_path: Path, quality: int = 95):
    """
    Save an image preserving its format with appropriate settings.
    """
    ext = dest_path.suffix.lower()
    
    if ext in ('.jpg', '.jpeg'):
        image.save(dest_path, quality=quality, optimize=True)
    elif ext == '.webp':
        image.save(dest_path, quality=quality, method=6)
    elif ext == '.png':
        image.save(dest_path, optimize=True)
    else:
        # For other formats, just save normally
        image.save(dest_path)


def resize_existing(output_dir: Path, max_size: int, quality: int = 95):
    """
    Resize existing images in output directory to meet max_size constraint.
    Deletes corrupt images. Skips images already within limits.
    """
    subdirs = ["domain0", "domain1", "uncertain", "ignored"]
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
    
    stats = {
        "resized": 0,
        "skipped": 0,
        "deleted_corrupt": 0,
    }
    
    all_images = []
    for subdir in subdirs:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            for file_path in subdir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    all_images.append(file_path)
    
    print(f"{C.BLUE}▶ Found {len(all_images)} images to check{C.RESET}")
    
    for file_path in tqdm(all_images, desc="Checking images", unit="img"):
        try:
            image = Image.open(file_path)
            image.load()  # Force load to detect corrupt images
            image = image.convert("RGB")
            
            w, h = image.size
            max_dim = max(w, h)
            
            if max_dim <= max_size:
                stats["skipped"] += 1
                continue
            
            # Resize needed
            resized = resize_image_to_max_size(image, max_size)
            save_image_with_format(resized, file_path, quality)
            stats["resized"] += 1
            
        except Exception as e:
            tqdm.write(f"{C.RED}✗ Corrupt image deleted: {file_path} ({e}){C.RESET}")
            file_path.unlink()
            stats["deleted_corrupt"] += 1
    
    print()
    print(f"{C.GREEN}{'='*50}{C.RESET}")
    print(f"{C.GREEN}Resize Existing Complete{C.RESET}")
    print(f"{C.GREEN}{'='*50}{C.RESET}")
    print(f"  Resized:         {stats['resized']:>6}")
    print(f"  Already OK:      {stats['skipped']:>6}")
    print(f"  Deleted corrupt: {stats['deleted_corrupt']:>6}")
    print(f"{C.GREEN}{'='*50}{C.RESET}")


def load_image_for_inference(path: str, max_size: int) -> torch.Tensor:
    """
    Load and preprocess an image for inference.
    Center crops to square, resizes, and normalizes to [-1, 1].
    """
    image = Image.open(path).convert("RGB")
    w, h = image.size
    
    # Center crop to square
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    
    # Resize to model size
    image = image.resize((max_size, max_size), Image.LANCZOS)
    
    # Normalize to [-1, 1]
    image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    image_array = np.transpose(image_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    
    return torch.from_numpy(image_array)


def classify_batch(
    model: RFPix2pixModel,
    images: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classify a batch of images using the saliency network.
    
    Returns:
        predictions: (B,) tensor of predicted domains (0 or 1)
        confidences: (B,) tensor of confidence scores for the predictions
    """
    with torch.no_grad():
        logits = model.get_saliency(images)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
    
    return predictions, confidences


def main():
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output)
    subdirs = {
        "domain0": output_dir / "domain0",
        "domain1": output_dir / "domain1",
        "uncertain": output_dir / "uncertain",
        "ignored": output_dir / "ignored",
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"{C.BLUE}▶ Output directory: {C.BOLD}{output_dir}{C.RESET}")
    
    # Handle resize-existing mode
    if args.resize_existing:
        if args.max_size is None:
            print(f"{C.RED}✗ --max-size is required with --resize-existing{C.RESET}")
            return
        print(f"{C.BLUE}▶ Resizing existing images to max {args.max_size}px{C.RESET}")
        resize_existing(output_dir, args.max_size, args.quality)
        return
    
    # Load model
    print(f"{C.BLUE}▶ Loading checkpoint: {C.BOLD}{args.checkpoint}{C.RESET}")
    model: RFPix2pixModel = load_module(args.checkpoint).to(device)  # type: ignore
    model.eval_saliency()
    max_size = model.max_size
    print(f"{C.GREEN}✓ Model loaded (image size: {max_size}x{max_size}){C.RESET}")
    
    # Get existing hashes to skip
    existing_hashes = get_existing_hashes(output_dir)
    print(f"{C.BLUE}▶ Found {len(existing_hashes)} existing files in output directories{C.RESET}")
    
    # Progressive image scanner
    print(f"{C.BLUE}▶ Scanning input directories progressively...{C.RESET}")
    scanner = ProgressiveImageScanner(args.input)
    
    # Stats
    stats = {
        "domain0": 0,
        "domain1": 0,
        "uncertain": 0,
        "skipped_duplicate": 0,
        "skipped_error": 0,
    }
    
    # Process images
    batch_paths: List[str] = []
    batch_hashes: List[str] = []
    batch_images: List[torch.Tensor] = []
    
    def process_batch():
        """Process the current batch of images."""
        if not batch_images:
            return
        
        # Stack and move to device
        images = torch.stack(batch_images).to(device)
        
        # Classify
        predictions, confidences = classify_batch(model, images)
        predictions = predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        # Copy files to appropriate directories
        for path, hash_val, pred, conf in zip(batch_paths, batch_hashes, predictions, confidences):
            ext = Path(path).suffix.lower()
            new_filename = f"{hash_val}{ext}"
            
            if conf < args.threshold:
                dest_dir = subdirs["uncertain"]
                stats["uncertain"] += 1
            elif pred == 0:
                dest_dir = subdirs["domain0"]
                stats["domain0"] += 1
            else:
                dest_dir = subdirs["domain1"]
                stats["domain1"] += 1
            
            dest_path = dest_dir / new_filename
            
            # Save with optional resizing
            if args.max_size is not None:
                try:
                    image = Image.open(path).convert("RGB")
                    image = resize_image_to_max_size(image, args.max_size)
                    save_image_with_format(image, dest_path, args.quality)
                except Exception as e:
                    tqdm.write(f"{C.RED}✗ Corrupt image skipped: {path} ({e}){C.RESET}")
                    # Remove from output if it was partially written
                    if dest_path.exists():
                        dest_path.unlink()
                    continue
            else:
                shutil.copy2(path, dest_path)
        
        # Clear batch
        batch_paths.clear()
        batch_hashes.clear()
        batch_images.clear()
    
    # Process all images with progress bar
    with tqdm(scanner, desc="Processing", unit="img") as pbar:
        for path in pbar:
            try:
                # Compute hash
                file_hash = compute_file_md5(path)
                
                # Skip if already processed or ignored
                if file_hash in existing_hashes:
                    stats["skipped_duplicate"] += 1
                    continue
                
                # Add to existing hashes to avoid duplicates within this run
                existing_hashes.add(file_hash)
                
                # Load image
                image = load_image_for_inference(path, max_size)
                
                # Add to batch
                batch_paths.append(path)
                batch_hashes.append(file_hash)
                batch_images.append(image)
                
                # Process batch if full
                if len(batch_images) >= args.batch_size:
                    process_batch()
                    
                    # Update progress bar description with stats
                    pbar.set_postfix({
                        "d0": stats["domain0"],
                        "d1": stats["domain1"],
                        "unc": stats["uncertain"],
                        "skip": stats["skipped_duplicate"],
                        "scan": scanner.status,
                    })
            
            except Exception as e:
                stats["skipped_error"] += 1
                tqdm.write(f"{C.RED}✗ Error processing {path}: {e}{C.RESET}")
    
    # Process remaining images in the last batch
    process_batch()
    
    # Print summary
    print()
    print(f"{C.GREEN}{'='*50}{C.RESET}")
    print(f"{C.GREEN}Processing Complete{C.RESET}")
    print(f"{C.GREEN}{'='*50}{C.RESET}")
    print(f"  Domain 0:    {stats['domain0']:>6}")
    print(f"  Domain 1:    {stats['domain1']:>6}")
    print(f"  Uncertain:   {stats['uncertain']:>6}")
    print(f"  Skipped (dup): {stats['skipped_duplicate']:>4}")
    print(f"  Skipped (err): {stats['skipped_error']:>4}")
    print(f"{C.GREEN}{'='*50}{C.RESET}")
    total_classified = stats['domain0'] + stats['domain1'] + stats['uncertain']
    print(f"  Total classified: {total_classified}")
    print()


if __name__ == "__main__":
    main()
