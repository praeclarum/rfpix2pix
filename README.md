# RF Pix2Pix

A neural network for **unpaired image-to-image translation** using Rectified Flow.

## Overview

RF Pix2Pix learns to translate images between two domains (e.g., horses ↔ zebras, day ↔ night) without requiring paired training data. Unlike traditional pix2pix which needs aligned image pairs, this approach works with separate collections of images from each domain.

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
pip install torch torchvision pillow numpy
```

## Usage

### Training

```bash
# Train a new model
python train.py --config configs/small.json \
    --domain0 path/to/domain0/images \
    --domain1 path/to/domain1/images

# Resume from checkpoint
python train.py --config configs/small.json \
    --domain0 path/to/domain0/images \
    --domain1 path/to/domain1/images \
    --checkpoint runs/run_xxx/model_1000.ckpt

# Development mode (smaller batches, more logging)
python train.py --config configs/small.json \
    --domain0 path/to/domain0/images \
    --domain1 path/to/domain1/images \
    --dev
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--config`, `-c` | Path to model config JSON file (required) |
| `--domain0`, `-d0` | Path(s) to domain 0 image directories (required) |
| `--domain1`, `-d1` | Path(s) to domain 1 image directories (required) |
| `--checkpoint`, `-ckpt` | Path to checkpoint file to resume training |
| `--run-id` | Custom run ID for output directory |
| `--dev` | Enable development mode |

## Project Structure

```
rfpix2pix/
├── train.py          # Training script and CLI
├── model.py          # RFPix2pixModel with saliency and velocity networks
├── data.py           # Dataset for loading unpaired image domains
├── fnn.py            # Neural network utilities and config system
└── configs/
    └── small.json    # Example configuration for small model
```

## Configuration

Models are configured via JSON files. Example (`configs/small.json`):

```json
{
    "type": "RFPix2pixModel",
    "max_size": 128,
    "sample_batch_size": 4,
    "train_batch_size": 8,
    "train_minibatch_size": 4,
    "train_images": 100000,
    "learning_rate": 0.0002,
    "num_inference_steps": 12,
    "timestep_sampling": "logit-normal",
    "saliency_accuracy_threshold": 0.99,
    "velocity_net": {
        "type": "UNet",
        "model_channels": 64,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2
    }
}
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `max_size` | Image resolution (images are resized to this) |
| `train_images` | Total images to train on |
| `learning_rate` | Optimizer learning rate |
| `num_inference_steps` | ODE integration steps during inference |
| `saliency_accuracy_threshold` | Required classifier accuracy before velocity training |
| `structure_pairing` | Enable structure-aware image pairing (default: false) |
| `structure_candidates` | Number of similar images to consider for pairing (default: 8) |

## Training Phases

1. **Phase 1: Saliency Training**  
   Trains the classifier to distinguish domain 0 from domain 1 until it reaches the accuracy threshold.

2. **Phase 1.5: Structure Embedding** (if `structure_pairing` is enabled)  
   Computes DINOv2 embeddings for all images in both domains. Embeddings are cached globally in `~/.cache/rfpix2pix/` to avoid recomputation across runs.

3. **Phase 2: Velocity Training**  
   Freezes the saliency network and trains the velocity field using the saliency-weighted loss. If structure pairing is enabled, domain-1 images are selected from the top `structure_candidates` most structurally similar images for each domain-0 sample.

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

## License

See [LICENSE](LICENSE) for details.

## References

- Liu, X., Gong, C., & Liu, Q. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
