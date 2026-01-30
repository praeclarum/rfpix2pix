from typing import List, Callable
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def get_image_paths(directories: List[str], extensions: List[str] = ['png', 'jpg', 'jpeg']) -> List[str]:
    paths = []
    for directory in directories:
        for ext in extensions:
            paths.extend(glob.glob(f"{directory}/**/*.{ext}", recursive=True))
    paths.sort()
    return paths

class RFPix2pixDataset(Dataset):
    def __init__(self, domain_0_paths: list[str], domain_1_paths: list[str], max_size: int, num_downsamples: int):
        self.domain_0_image_paths = get_image_paths(domain_0_paths)
        self.domain_1_image_paths = get_image_paths(domain_1_paths)
        self.max_image_size = max_size
        self.image_size_multiple = 2 ** num_downsamples

    def __len__(self):
        return max(len(self.domain_0_image_paths), len(self.domain_1_image_paths))

    def __getitem__(self, index):
        domain_0_path = self.domain_0_image_paths[random.randint(0, len(self.domain_0_image_paths) - 1)]
        domain_1_path = self.domain_1_image_paths[random.randint(0, len(self.domain_1_image_paths) - 1)]
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
