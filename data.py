from typing import List, Callable, Optional, Tuple
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from PIL import Image

def get_image_paths(directories: List[str], extensions: List[str] = ['png', 'jpg', 'jpeg']) -> List[str]:
    paths = []
    for directory in directories:
        for ext in extensions:
            paths.extend(glob.glob(f"{directory}/**/*.{ext}", recursive=True))
    paths.sort()
    return paths


class StructurePairing:
    """
    Structure-aware pairing for velocity training.
    
    Given precomputed structure embeddings for both domains, finds
    structurally similar images across domains using cosine similarity.
    Enables better velocity training by pairing images with similar
    composition, pose, and layout.
    """
    
    def __init__(
        self,
        embeddings_0: np.ndarray,
        embeddings_1: np.ndarray,
        structure_candidates: int = 8,
    ):
        """
        Initialize structure pairing.
        
        Args:
            embeddings_0: (N, D) normalized embeddings for domain 0 images
            embeddings_1: (M, D) normalized embeddings for domain 1 images
            structure_candidates: Number of top similar images to consider
        """
        self.embeddings_0 = embeddings_0  # (N, D)
        self.embeddings_1 = embeddings_1  # (M, D)
        self.structure_candidates = structure_candidates
        
        # Precompute similarity matrix: (N, M) = embeddings_0 @ embeddings_1.T
        # Since embeddings are L2-normalized, dot product = cosine similarity
        self.similarity_matrix = embeddings_0 @ embeddings_1.T  # (N, M)
        
        # Precompute top-K indices for each domain 0 image
        # This avoids repeated argsort during training
        k = min(structure_candidates, self.similarity_matrix.shape[1])
        # argsort in descending order, take top k
        self.top_k_indices = np.argsort(-self.similarity_matrix, axis=1)[:, :k]  # (N, k)
    
    def get_paired_index(self, domain_0_idx: int) -> int:
        """
        Get a structurally similar domain 1 index for a domain 0 image.
        
        Randomly samples from top-K most similar domain 1 images.
        
        Args:
            domain_0_idx: Index of the domain 0 image
            
        Returns:
            Index of a structurally similar domain 1 image
        """
        candidates = self.top_k_indices[domain_0_idx]
        return int(random.choice(candidates))


class RFPix2pixDataset(Dataset):
    def __init__(
        self,
        domain_0_paths: list[str],
        domain_1_paths: list[str],
        max_size: int,
        num_downsamples: int,
        structure_pairing: Optional[StructurePairing] = None,
    ):
        self.domain_0_image_paths = get_image_paths(domain_0_paths)
        self.domain_1_image_paths = get_image_paths(domain_1_paths)
        self.max_image_size = max_size
        self.image_size_multiple = 2 ** num_downsamples
        self.structure_pairing = structure_pairing
        
        # Validate structure pairing dimensions match
        if structure_pairing is not None:
            n0 = len(self.domain_0_image_paths)
            n1 = len(self.domain_1_image_paths)
            e0 = structure_pairing.embeddings_0.shape[0]
            e1 = structure_pairing.embeddings_1.shape[0]
            if e0 != n0 or e1 != n1:
                raise ValueError(
                    f"Structure pairing dimension mismatch: "
                    f"embeddings ({e0}, {e1}) != paths ({n0}, {n1})"
                )

    def __len__(self):
        return max(len(self.domain_0_image_paths), len(self.domain_1_image_paths))

    def __getitem__(self, index):
        # Sample domain 0 image (random)
        domain_0_idx = random.randint(0, len(self.domain_0_image_paths) - 1)
        domain_0_path = self.domain_0_image_paths[domain_0_idx]
        
        # Sample domain 1 image (structure-paired or random)
        if self.structure_pairing is not None:
            domain_1_idx = self.structure_pairing.get_paired_index(domain_0_idx)
        else:
            domain_1_idx = random.randint(0, len(self.domain_1_image_paths) - 1)
        domain_1_path = self.domain_1_image_paths[domain_1_idx]
        
        domain_0_image = self.load_and_preprocess_image(domain_0_path)
        domain_1_image = self.load_and_preprocess_image(domain_1_path)
        return {'domain_0': domain_0_image, 'domain_1': domain_1_image}
    
    def load_and_preprocess_image(self, path: str) -> torch.Tensor:
        """Load an image and preprocess it to fit the model requirements."""
        image = Image.open(path).convert("RGB")
        src_width, src_height = image.size
        prescale = 0.8 + 0.2 * random.random()
        if src_width >= src_height:
            crop_size = int(src_height * prescale)
        else:
            crop_size = int(src_width * prescale)
        crop_x = random.randint(0, src_width - crop_size)
        crop_y = random.randint(0, src_height - crop_size)
        image = image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        image = image.resize((self.max_image_size, self.max_image_size), Image.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_tensor = torch.from_numpy(image_array)
        # Random horizontal flip
        should_hflip = random.random() < 0.5
        if should_hflip:
            image_tensor = torch.flip(image_tensor, dims=[2])
        return image_tensor

class SaliencyAugmentation(nn.Module):
    """
    Data augmentation for saliency network training.
    
    Wraps torchvision transforms to work with [-1, 1] colorspace.
    Converts to [0, 1] before transforms, then back to [-1, 1] after.
    
    Supported augmentations:
        - "color_jitter": Random brightness/contrast/saturation/hue
        - "grayscale": Random grayscale conversion (p=0.1)
        - "random_erasing": Random rectangular patch erasure
        - "gaussian_blur": Random Gaussian blur
    """
    
    def __init__(self, augmentations: List[str]):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of augmentation names to apply.
                          Empty list = no augmentation (identity).
        """
        super().__init__()
        self.augmentations = augmentations
        
        if not augmentations:
            self.transform = None
            return
        
        transforms = []
        for aug in augmentations:
            if aug == "color_jitter":
                # Random brightness/contrast/saturation/hue
                # Forces learning texture/shape over color shortcuts
                transforms.append(T.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                ))
            elif aug == "grayscale":
                # Random grayscale with low probability
                # Useful if color isn't the primary style difference
                transforms.append(T.RandomGrayscale(p=0.1))
            elif aug == "random_erasing":
                # Random rectangular patch erasure
                # Prevents relying on single discriminative regions
                # Note: operates on [0,1] range, erases with random values
                transforms.append(T.RandomErasing(
                    p=0.2,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                ))
            elif aug == "gaussian_blur":
                # Random Gaussian blur
                # Forces learning coarser structure
                transforms.append(T.GaussianBlur(
                    kernel_size=5,
                    sigma=(0.1, 2.0),
                ))
            else:
                raise ValueError(f"Unknown saliency augmentation: {aug}")
        
        self.transform = T.Compose(transforms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to input tensor.
        
        Args:
            x: (B, 3, H, W) images in [-1, 1] range
            
        Returns:
            (B, 3, H, W) augmented images in [-1, 1] range
        """
        if self.transform is None:
            return x
        
        # Convert [-1, 1] -> [0, 1] for torchvision transforms
        x_01 = (x + 1.0) / 2.0
        
        # Apply transforms
        x_aug = self.transform(x_01)
        
        # Convert [0, 1] -> [-1, 1]
        x_out = x_aug * 2.0 - 1.0
        
        return x_out

