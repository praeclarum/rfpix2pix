# RF Pix2Pix

A neural network for **unpaired image-to-image translation** and **image generation** using Rectified Flow.

## Overview

RF Pix2Pix learns to translate images between two domains (e.g., horses ↔ zebras, day ↔ night) without requiring paired training data. Unlike traditional pix2pix which needs aligned image pairs, this approach works with separate collections of images from each domain.

The model also supports **generative mode**, which creates images from random noise similar to diffusion models, providing a unified framework for both domain transfer and generation tasks.

The method is based on Section 5.3 of the paper:

> **"Flow Straight and Fast: Learning to Generate Data and Transfer Data with Rectified Flow"**  
> by Xingchao Liu, Chengyue Gong, and Qiang Liu

## How It Works

The model learns a **velocity field** that transforms images from one domain to another using rectified flow concepts:

1. **Saliency Network**: A classifier trained to distinguish between the two image domains. Its gradients serve as a "saliency score" that identifies which image features are important for domain classification.

2. **Velocity Network**: A UNet that learns to predict the velocity vector `v(Z_t, t)` for transforming images along the flow path.

The key insight is that by minimizing:

$$\min_v \int_0^1 \mathbb{E}\left[\left\|\nabla h(X_t)^\top (X_1 - X_0 - v(X_t, t))\right\|^2\right] dt$$

where `h(x)` is the saliency network's feature representation, the model learns to focus on transferring style-relevant features while preserving content.

## Installation

```bash
# Clone the repository
git clone https://github.com/praeclarum/rfpix2pix.git
cd rfpix2pix

# Install dependencies
pip install torch torchvision pillow numpy wandb
```

## Usage

### Modes

RF Pix2Pix supports two distinct modes:

**1. Image Translation Mode (Unpaired Domain Transfer)**

Translates images between two domains using unpaired data (e.g., horses ↔ zebras, sketches ↔ photos). Requires separate collections of images from both domains. Uses a saliency network to guide the translation.

```bash
python train.py --config configs/small.json \
    --domain0 path/to/horses \
    --domain1 path/to/zebras
```

**2. Generative Mode (Noise-to-Image)**

Generates images from random noise, similar to standard diffusion models. Set domain 0 to the special `"random"` keyword to sample from standard Gaussian noise N(0,1). The saliency network is bypassed and training uses simple MSE loss.

```bash
python train.py --config configs/small.json \
    --domain0 random \
    --domain1 path/to/faces
```

### Training Examples

```bash
# Image translation: horses to zebras
python train.py --config configs/small.json \
    --domain0 path/to/horses \
    --domain1 path/to/zebras

# Generative: create faces from noise
python train.py --config configs/small.json \
    --domain0 random \
    --domain1 path/to/faces

# Resume from checkpoint
python train.py --config configs/small.json \
    --domain0 path/to/domain0 \
    --domain1 path/to/domain1 \
    --checkpoint runs/run_xxx/model_1000.ckpt

# Development mode (smaller batches, more logging)
python train.py --config configs/small.json \
    --domain0 path/to/domain0 \
    --domain1 path/to/domain1 \
    --dev
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--config`, `-c` | Path to model config JSON file (required) |
| `--domain0`, `-d0` | Path(s) to domain 0 media directories, or `"random"` for generative mode |
| `--domain1`, `-d1` | Path(s) to domain 1 media directories (images and/or videos) |
| `--checkpoint`, `-ckpt` | Path to checkpoint file to resume training |
| `--run-id` | Custom run ID for output directory |
| `--dev` | Enable development mode |

### Video Sources

- Domain directories can mix still images (`.png`, `.jpg`, `.jpeg`) and video clips (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`).
- Videos are expanded into individual frame samples. A 1-minute clip at 30 fps therefore yields ~1,800 training examples and structure-pairing targets, which keeps multi-scene footage usable and makes per-frame similarity meaningful.
- Each frame’s cache key is derived from the MD5 of the underlying video plus the frame index, so DINO embeddings are stored per-frame and remain stable across runs.
- The dataloader still applies the same crop/resize/[-1,1] preprocessing to both images and video frames, so augmentations and normalization stay consistent.
- Expect structure-pairing preparation to take longer on long-form footage, since every frame is hashed and embedded exactly once before training.
- Video decoding relies on `torchvision.io.VideoReader`, so ensure your TorchVision build includes FFmpeg support (the default pip wheels do). No extra dependencies are required beyond `torch`, `torchvision`, `pillow`, and `numpy`.

## Project Structure

```
rfpix2pix/
├── codec.py          # Codec network to convert between pixel and latent space
├── data.py           # Dataset for loading unpaired image/video domains
├── fnn.py            # Neural network utilities and config system
├── model.py          # RFPix2pixModel and Velocity networks
├── saliency.py       # Saliency network and related losses
├── split.py          # Automatic domain0/domain1 splitting from uncategorized images
├── train_codec.py    # Script for training just the codec
├── train_velocity.py # Script for training just the velocity network
├── train_saliency.py # Script for training just the saliency network
├── train.py          # Training script and CLI
├── utils.py          # Junk drawer
└── configs/          # Example configurations
```

## Configuration

Models are configured via JSON files. Example (`configs/small.json`):

```json
{
    "type": "RFPix2pixModel",
    "max_size": 128,
    "sample_batch_size": 8,
    "train_batch_size": 48,
    "train_minibatch_size": 24,
    "train_images": 6000000,
    "learning_rate": 0.0002,
    "num_inference_steps": 12,
    "timestep_sampling": "logit-normal",
    "saliency_learning_rate": 0.00005,
    "saliency_accuracy_threshold": 0.995,
    "saliency_warmup_threshold": 0.90,
    "saliency_blend_fraction": 0.0,
    "saliency_label_smoothing": 0.1,
    "saliency_augmentations": ["color_jitter", "grayscale", "hflip", "random_erasing"],
    "structure_pairing": false,
    "structure_candidates": 8,
    "velocity_net": {
        "type": "UNet",
        "model_channels": 128,
        "ch_mult": [1, 2, 4, 8, 8],
        "normalization": "GroupNorm32",
        "activation": "SiLU",
        "num_res_blocks": 2,
        "zero_res_blocks": false,
        "attention_resolutions": [],
        "num_attention_heads": 8
    },
    "saliency_net": {
        "type": "ResNetSaliencyNet",
        "backbone": "resnet50",
        "num_classes": 2,
        "pretrained": true,
        "latent_channels": 512
    }
}
```

## Training Phases

**Image Translation Mode:**

1. **Phase 1: Saliency Training**  
   Trains the classifier to distinguish domain 0 from domain 1 until it reaches the accuracy threshold.

2. **Phase 1.5: Structure Embedding** (if `structure_pairing` is enabled)  
   Computes DINOv2 embeddings for all images in both domains. Embeddings are cached globally in `~/.cache/rfpix2pix/` to avoid recomputation across runs.

3. **Phase 2: Velocity Training**  
   Freezes the saliency network and trains the velocity field using the saliency-weighted loss. If structure pairing is enabled, domain-1 images are selected from the top `structure_candidates` most structurally similar images for each domain-0 sample.

**Generative Mode:**

In generative mode (`--domain0 random`), Phase 1 and 1.5 are skipped entirely. The model proceeds directly to velocity training using simple MSE loss without saliency weighting.

### Structure-Aware Pairing

For datasets with diverse compositions (e.g., close-ups vs wide shots), enabling `structure_pairing` can improve training by matching images with similar layouts across domains. This uses DINOv2 embeddings to find structurally similar images, avoiding mismatched pairs like a horse close-up being paired with a distant zebra.

```json
{
    "structure_pairing": true,
    "structure_candidates": 16
}
```

Higher `structure_candidates` values provide more variety but less strict matching. Lower values enforce stricter structural similarity.

## Hardware Support

- **CUDA**: Automatic GPU acceleration on NVIDIA GPUs
- **MPS**: Apple Silicon GPU support
- **CPU**: Fallback when no GPU is available

## Future Improvements

### FiLM Conditioning in ResBlocks

The current UNet injects timestep information by adding a projected embedding **after** each ResBlock stack:

```
h = ResBlock(h)
h = h + Linear(t_emb)  # additive bias, applied after the block
```

This means the ResBlock's internal convolutions operate without any awareness of the current timestep. The time signal only modulates the output as a spatially-uniform bias.

Stable Diffusion's UNet uses **FiLM conditioning** (Feature-wise Linear Modulation), also called Adaptive Group Normalization (AdaGN). The timestep embedding is injected *inside* each ResBlock, modulating the GroupNorm parameters between the two convolution layers:

```
h = Conv(norm(h))             # first conv with pre-norm
scale, shift = Linear(t_emb)  # project time to per-channel scale + shift
h = scale * GroupNorm(h) + shift  # modulate normalization (FiLM)
h = Conv(h)                   # second conv
h = h + skip                  # residual
```

**Why it matters:** FiLM gives each layer two degrees of freedom per channel (scale and shift) instead of one (bias only). The scale component is particularly important — it allows the network to amplify or suppress individual feature channels based on the timestep. At early flow times (t≈0, near the source domain / noise), the network may need very different feature responses than at late times (t≈1, near the target domain). Additive-only conditioning can adjust feature *levels* but cannot *gate* features on or off.

**Implementation cost:** This requires modifying `ResBlock` to accept a conditioning vector and splitting its `forward()` into two halves around the modulated norm. Since `ResBlock` is shared across the entire codebase (UNet, Encoder, Decoder, codec nets), the change must be backward-compatible — the conditioning input should be optional, defaulting to the current unconditional behavior.

## License

See [LICENSE](LICENSE) for details.

## References

- Liu, X., Gong, C., & Liu, Q. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
