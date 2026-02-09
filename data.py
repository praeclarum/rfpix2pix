from typing import Dict, List, Callable, Optional, Tuple
import os
import gc
import glob
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.io import VideoReader
from PIL import Image
import sqlite3
from pathlib import Path

from tqdm import tqdm

from utils import compute_file_md5, Colors as C


def _normalize_extension(ext: str) -> str:
    return ext.lower().lstrip('.')


IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg']
VIDEO_EXTENSIONS = ['mp4', 'mov', 'm4v', 'avi', 'mkv', 'webm']
IMAGE_EXTENSION_SET = {_normalize_extension(ext) for ext in IMAGE_EXTENSIONS}
VIDEO_EXTENSION_SET = {_normalize_extension(ext) for ext in VIDEO_EXTENSIONS}
MEDIA_EXTENSION_SET = IMAGE_EXTENSION_SET | VIDEO_EXTENSION_SET


def is_video_file(path: str) -> bool:
    suffix = Path(path).suffix
    if not suffix:
        return False
    return _normalize_extension(suffix) in VIDEO_EXTENSION_SET

def get_image_paths(directories: List[str], extensions: Optional[List[str]] = None) -> List[str]:
    """Collect paths to supported media (images and videos)."""
    normalized_exts = sorted({_normalize_extension(ext) for ext in (extensions or MEDIA_EXTENSION_SET)})
    paths = []
    for directory in directories:
        # Skip magic "random" string for generative mode
        if directory.lower() == "random":
            continue
        for ext in normalized_exts:
            paths.extend(glob.glob(f"{directory}/**/*.{ext}", recursive=True))
    paths.sort()
    return paths

def _frame_tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3:
        raise ValueError(f"Unexpected video frame rank: {frame.shape}")
    tensor = frame.cpu()
    if tensor.shape[0] <= 4:  # Likely CHW
        tensor = tensor.permute(1, 2, 0)
    arr = tensor.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr).convert("RGB")


class VideoFrameSampler:
    """Utility for fast random access into video files."""

    def __init__(self, cache_size: int = 4):
        self.cache_size = cache_size
        self._reader_cache: "OrderedDict[str, VideoReader]" = OrderedDict()
        self._metadata_cache: Dict[str, Dict[str, float]] = {}

    def _evict_oldest(self):
        if not self._reader_cache:
            return
        old_path, _ = self._reader_cache.popitem(last=False)
        self._metadata_cache.pop(old_path, None)

    def _get_reader(self, path: str) -> VideoReader:
        reader = self._reader_cache.get(path)
        if reader is not None:
            self._reader_cache.move_to_end(path)
            return reader
        try:
            reader = VideoReader(path, "video")
        except (RuntimeError, OSError) as exc:
            raise RuntimeError(
                f"Failed to open video '{path}'. Ensure torchvision is built with FFmpeg support."
            ) from exc
        self._reader_cache[path] = reader
        if len(self._reader_cache) > self.cache_size:
            self._evict_oldest()
        return reader

    def _get_metadata(self, path: str, reader: VideoReader) -> Dict[str, float]:
        meta = self._metadata_cache.get(path)
        if meta is not None:
            return meta
        raw_meta = reader.get_metadata().get("video", {})
        duration = raw_meta.get("duration")
        fps = raw_meta.get("fps")
        frames = raw_meta.get("frames")
        if duration is None and frames is not None and fps not in (None, 0):
            duration = frames / fps
        if duration is None:
            duration = float(frames or 1) / float(fps or 30.0)
        duration = max(duration, 1e-3)
        meta = {"duration": duration}
        self._metadata_cache[path] = meta
        return meta

    def _timestamp(self, selection: str, metadata: Dict[str, float]) -> float:
        duration = metadata["duration"]
        if selection == "random":
            return random.uniform(0.0, max(duration - 1e-3, 0.0))
        if selection == "middle":
            return min(duration * 0.5, max(duration - 1e-3, 0.0))
        raise ValueError(f"Unknown frame selection '{selection}'")

    def get_frame(self, path: str, selection: str = "random") -> Image.Image:
        reader = self._get_reader(path)
        metadata = self._get_metadata(path, reader)
        timestamp = self._timestamp(selection, metadata)
        for _ in range(5):
            reader.seek(max(timestamp, 0.0))
            try:
                frame = next(reader)["data"]
                return _frame_tensor_to_pil(frame)
            except StopIteration:
                timestamp = self._timestamp("random", metadata)
        reader.seek(0.0)
        try:
            frame = next(reader)["data"]
        except StopIteration as exc:
            raise RuntimeError(f"Unable to decode frames from video '{path}'") from exc
        return _frame_tensor_to_pil(frame)


DEFAULT_VIDEO_SAMPLER: Optional[VideoFrameSampler] = None


def get_default_video_sampler() -> VideoFrameSampler:
    global DEFAULT_VIDEO_SAMPLER
    if DEFAULT_VIDEO_SAMPLER is None:
        DEFAULT_VIDEO_SAMPLER = VideoFrameSampler()
    return DEFAULT_VIDEO_SAMPLER


def preprocess_pil_image(image: Image.Image, max_size: int) -> torch.Tensor:
    src_width, src_height = image.size
    prescale = 0.9 + 0.1 * random.random()
    if src_width >= src_height:
        haspect = min(4 / 3, src_width / src_height)
        crop_height = int(src_height * prescale)
        crop_width = int(crop_height * haspect)
        if crop_width > src_width:
            crop_width = src_width
            crop_height = int(crop_width / haspect)
    else:
        vaspect = min(4 / 3, src_height / src_width)
        crop_width = int(src_width * prescale)
        crop_height = int(crop_width * vaspect)
        if crop_height > src_height:
            crop_height = src_height
            crop_width = int(crop_height / vaspect)
    crop_x = (src_width - crop_width) // 2
    crop_y = (src_height - crop_height) // 2
    image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    image = image.resize((max_size, max_size), Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array)


def load_and_preprocess_image(
    path: str,
    max_size: int,
    video_sampler: Optional[VideoFrameSampler] = None,
    frame_selection: str = "random",
) -> torch.Tensor:
    """Load an image or video frame and preprocess it to fit model requirements.

    Args:
        path: Path to an image or supported video file.
        max_size: Target square resolution.
        video_sampler: Optional sampler to reuse cached VideoReader instances.
        frame_selection: "random" for dataloader parity or "middle" for deterministic frames.
    """
    if is_video_file(path):
        sampler = video_sampler or get_default_video_sampler()
        pil_image = sampler.get_frame(path, selection=frame_selection)
    else:
        with Image.open(path) as img:
            pil_image = img.convert("RGB")
    return preprocess_pil_image(pil_image, max_size)

class RFPix2pixDataset(Dataset):
    def __init__(
        self,
        domain_0_paths: list[str],
        domain_1_paths: list[str],
        max_size: int,
        num_downsamples: int,
        structure_pairing: Optional["StructurePairing"] = None,
    ):
        self.domain_0_image_paths = get_image_paths(domain_0_paths)
        self.domain_1_image_paths = get_image_paths(domain_1_paths)
        self.max_image_size = max_size
        self.image_size_multiple = 2 ** num_downsamples
        self.structure_pairing = structure_pairing
        
        # Detect generative mode (random noise for domain 0)
        self.use_random_noise_domain0 = len(self.domain_0_image_paths) == 0
        
        # Warn if structure pairing is used with random domain 0
        if self.use_random_noise_domain0 and structure_pairing is not None:
            print(f"{C.YELLOW}⚠ Structure pairing is enabled but domain 0 is random noise. "
                  f"Pairing will be ignored.{C.RESET}")
        
        # Validate structure pairing dimensions match
        if structure_pairing is not None and not self.use_random_noise_domain0:
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
        if self.use_random_noise_domain0:
            return len(self.domain_1_image_paths)
        return max(len(self.domain_0_image_paths), len(self.domain_1_image_paths))

    def __getitem__(self, index):
        # Sample domain 0 image (random noise in generative mode, otherwise loaded image)
        if self.use_random_noise_domain0:
            # Generative mode: use standard normal noise (unclipped)
            domain_0_image = torch.randn(3, self.max_image_size, self.max_image_size)
            domain_1_idx = random.randint(0, len(self.domain_1_image_paths) - 1)
        else:
            # Image translation mode: load images
            domain_0_idx = random.randint(0, len(self.domain_0_image_paths) - 1)
            domain_0_path = self.domain_0_image_paths[domain_0_idx]
            domain_0_image = load_and_preprocess_image(
                domain_0_path,
                self.max_image_size,
            )
            
            # Sample domain 1 image (structure-paired or random)
            if self.structure_pairing is not None:
                domain_1_idx = self.structure_pairing.get_paired_index(domain_0_idx)
            else:
                domain_1_idx = random.randint(0, len(self.domain_1_image_paths) - 1)
        
        domain_1_path = self.domain_1_image_paths[domain_1_idx]
        domain_1_image = load_and_preprocess_image(
            domain_1_path,
            self.max_image_size,
        )
        return {'domain_0': domain_0_image, 'domain_1': domain_1_image}
    

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
            elif aug == "hflip":
                # Random horizontal flip
                transforms.append(T.RandomHorizontalFlip(p=0.5))
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

# DINOv2 constants for embedding computation
# Use 224 (standard ViT size) for consistent embeddings across runs
DINO_TARGET_SIZE = 224
DINO_PATCH_SIZE = 14

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

class StructureEncoder(nn.Module):
    """
    Structure encoder using DINOv2 for domain-agnostic image embeddings.
    
    Used for structure-aware pairing: finds structurally similar images
    across domains for better velocity training. Unlike the saliency network
    which learns domain-discriminative features, DINOv2 provides semantic
    structure embeddings that capture composition, pose, and layout.
    
    The encoder is always frozen (pretrained weights only).
    """
    
    # DINOv2 model variants and their embedding dimensions
    DINO_MODELS = {
        "dinov2_vits14": 384,   # Small: 21M params
        "dinov2_vitb14": 768,   # Base: 86M params
        "dinov2_vitl14": 1024,  # Large: 300M params
    }
    
    # Type hints for registered buffers
    imagenet_mean: torch.Tensor
    imagenet_std: torch.Tensor
    
    def __init__(self, model_name: str = "dinov2_vits14"):
        """
        Initialize DINOv2 structure encoder.
        
        Args:
            model_name: DINOv2 variant to use. One of:
                - "dinov2_vits14" (default, 384-dim, fastest)
                - "dinov2_vitb14" (768-dim)
                - "dinov2_vitl14" (1024-dim, best quality)
        """
        super().__init__()
        
        if model_name not in self.DINO_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Supported: {list(self.DINO_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self._embedding_dim = self.DINO_MODELS[model_name]
        
        # Register ImageNet normalization constants as buffers
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        # Load DINOv2 model from torch.hub (downloads on first use)
        dino_model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
        )
        assert isinstance(dino_model, nn.Module)
        self.model: nn.Module = dino_model
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of the output embedding vectors."""
        return self._embedding_dim
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from app's [-1, 1] colorspace to ImageNet normalization."""
        x_01 = (x + 1.0) / 2.0
        x_normalized = (x_01 - self.imagenet_mean) / self.imagenet_std
        return x_normalized
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to structure embeddings.
        
        Args:
            x: (B, 3, H, W) input images in [-1, 1] range
            
        Returns:
            (B, embedding_dim) normalized embedding vectors
        """
        x_normalized = self._normalize_input(x)
        
        # DINOv2 returns CLS token embedding
        embeddings = self.model(x_normalized)  # (B, embedding_dim)
        
        # L2 normalize for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def get_cache_dir() -> Path:
    """Get the cache directory, respecting RFPIX2PIX_CACHE_DIR env var."""
    cache_dir = os.environ.get("RFPIX2PIX_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "rfpix2pix"


class EmbeddingCache:
    """
    SQLite-backed cache for image structure embeddings.
    
    Maps image MD5 hashes to embedding vectors. Thread-safe for reads,
    uses write-ahead logging for concurrent access across processes.
    """
    
    def __init__(self, db_name: str = "dino_embeddings.db"):
        """
        Initialize the embedding cache.
        
        Args:
            db_name: Name of the SQLite database file.
        """
        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / db_name
        
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    md5 TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            """)
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        finally:
            conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def get(self, md5: str) -> Optional[np.ndarray]:
        """
        Get an embedding by MD5 hash.
        
        Args:
            md5: MD5 hash of the image file.
            
        Returns:
            Embedding vector as numpy array, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE md5 = ?",
                (md5.lower(),)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return np.frombuffer(row[0], dtype=np.float32)
        finally:
            conn.close()
    
    def put(self, md5: str, embedding: np.ndarray):
        """
        Store an embedding.
        
        Args:
            md5: MD5 hash of the image file.
            embedding: Embedding vector as numpy array.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (md5, embedding) VALUES (?, ?)",
                (md5.lower(), embedding.astype(np.float32).tobytes())
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_batch(self, md5_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Get multiple embeddings by MD5 hash.
        
        Args:
            md5_list: List of MD5 hashes to look up.
            
        Returns:
            Dict mapping found MD5s to their embeddings.
        """
        if not md5_list:
            return {}
        
        conn = self._get_connection()
        try:
            # Use parameterized query with IN clause
            placeholders = ",".join("?" * len(md5_list))
            cursor = conn.execute(
                f"SELECT md5, embedding FROM embeddings WHERE md5 IN ({placeholders})",
                [m.lower() for m in md5_list]
            )
            result = {}
            for row in cursor:
                result[row[0]] = np.frombuffer(row[1], dtype=np.float32)
            return result
        finally:
            conn.close()
    
    def put_batch(self, embeddings: Dict[str, np.ndarray]):
        """
        Store multiple embeddings.
        
        Args:
            embeddings: Dict mapping MD5 hashes to embedding vectors.
        """
        if not embeddings:
            return
        
        conn = self._get_connection()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (md5, embedding) VALUES (?, ?)",
                [(md5.lower(), emb.astype(np.float32).tobytes()) for md5, emb in embeddings.items()]
            )
            conn.commit()
        finally:
            conn.close()
    
    def __len__(self) -> int:
        """Return the number of cached embeddings."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def __repr__(self) -> str:
        return f"EmbeddingCache({self.db_path}, {len(self)} entries)"

def compute_structure_embeddings(
    image_paths: list[str],
    encoder: StructureEncoder,
    cache: EmbeddingCache,
    device: torch.device,
    desc: str = "Computing embeddings",
) -> np.ndarray:
    """
    Compute structure embeddings for a list of media files (images or videos).

    Uses the embedding cache to avoid recomputing embeddings for inputs
    that have been processed in previous runs.
    
    Images are resized preserving aspect ratio (short side = 224), then
    minimally cropped to ensure dimensions are multiples of 14 for DINOv2.
    
    Args:
        image_paths: List of media file paths
        encoder: StructureEncoder (DINOv2) model
        cache: EmbeddingCache for persistent storage
        desc: Description for progress bar
        
    Returns:
        (N, D) array of embeddings, one per image path (in order)
    """
    import numpy as np
    from PIL import Image
    
    n_images = len(image_paths)
    embedding_dim = encoder.embedding_dim
    
    # Compute MD5 hashes for all images
    print(f"{C.DIM}  Computing file hashes...{C.RESET}")
    md5_hashes = [compute_file_md5(p) for p in tqdm(image_paths, desc="Hashing")]
    
    # Check cache for existing embeddings
    cached = cache.get_batch(md5_hashes)
    
    # Identify which images need computation
    needs_compute = []
    for i, md5 in enumerate(md5_hashes):
        if md5 not in cached:
            needs_compute.append(i)
    
    n_cached = n_images - len(needs_compute)
    n_to_compute = len(needs_compute)
    print(f"{C.DIM}  Found {C.CYAN}{n_cached}{C.RESET}{C.DIM} cached, need to compute {C.CYAN}{n_to_compute}{C.RESET}")
    
    # Compute missing embeddings
    video_sampler: Optional[VideoFrameSampler] = None

    if needs_compute:
        new_embeddings = {}
        
        # Process one image at a time (images have different aspect ratios)
        for idx in tqdm(needs_compute, desc=desc):
            path = image_paths[idx]
            if is_video_file(path):
                if video_sampler is None:
                    video_sampler = VideoFrameSampler(cache_size=2)
                img = video_sampler.get_frame(path, selection="middle")
            else:
                img = Image.open(path).convert("RGB")
            
            # Resize preserving aspect ratio (short side = DINO_TARGET_SIZE)
            w, h = img.size
            if w < h:
                new_w = DINO_TARGET_SIZE
                new_h = int(h * DINO_TARGET_SIZE / w)
            else:
                new_h = DINO_TARGET_SIZE
                new_w = int(w * DINO_TARGET_SIZE / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Minimally crop to make dimensions multiples of DINO_PATCH_SIZE
            w, h = img.size
            crop_w = (w // DINO_PATCH_SIZE) * DINO_PATCH_SIZE
            crop_h = (h // DINO_PATCH_SIZE) * DINO_PATCH_SIZE
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
            
            # Convert to tensor in [-1, 1]
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
            
            # Encode
            with torch.no_grad():
                embedding = encoder(tensor).cpu().numpy()[0]
            
            # Store in cache
            md5 = md5_hashes[idx]
            new_embeddings[md5] = embedding
        
        # Batch write to cache
        cache.put_batch(new_embeddings)
        
        # Merge with cached
        cached.update(new_embeddings)
    
    # Build output array in original order
    result = np.zeros((n_images, embedding_dim), dtype=np.float32)
    for i, md5 in enumerate(md5_hashes):
        result[i] = cached[md5]
    
    return result


def prepare_structure_pairing(
    dataset: RFPix2pixDataset,
    structure_candidates: int,
    device: torch.device,
) -> StructurePairing:
    """
    Prepare structure pairing by computing embeddings for both domains.
    
    Args:
        dataset: Dataset with image paths for both domains
        structure_candidates: Number of similar images to consider for pairing
        
    Returns:
        StructurePairing object ready for use in dataset
    """
    print(f"\n{C.BOLD}{C.MAGENTA}━━━ Computing Structure Embeddings ━━━{C.RESET}")
    
    # Initialize encoder and cache
    print(f"{C.DIM}  Loading DINOv2 encoder...{C.RESET}")
    encoder = StructureEncoder().to(device)
    cache = EmbeddingCache()
    print(f"{C.DIM}  Cache: {cache}{C.RESET}")
    
    # Compute embeddings for both domains
    print(f"\n{C.BRIGHT_CYAN}  Domain 0:{C.RESET} {len(dataset.domain_0_image_paths)} media files")
    embeddings_0 = compute_structure_embeddings(
        dataset.domain_0_image_paths,
        encoder,
        cache,
        device,
        desc="Domain 0",
    )
    
    print(f"\n{C.BRIGHT_CYAN}  Domain 1:{C.RESET} {len(dataset.domain_1_image_paths)} media files")
    embeddings_1 = compute_structure_embeddings(
        dataset.domain_1_image_paths,
        encoder,
        cache,
        device,
        desc="Domain 1",
    )
    
    # Create pairing object
    pairing = StructurePairing(
        embeddings_0=embeddings_0,
        embeddings_1=embeddings_1,
        structure_candidates=structure_candidates,
    )
    
    print(f"\n{C.GREEN}✓ Structure pairing ready{C.RESET} (top-{structure_candidates} candidates per image)\n")
    
    # Clean up encoder from GPU
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return pairing


