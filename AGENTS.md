# RF Pix2pix

This is a neural network specializing in image-to-image translation using unpaired data.

It is based on Section 5.3 of the paper "Flow Straight and Fast: Learning to Generate Data and Transfer Data with Rectified Flow" by Xingchao Liu, Chengyue Gong, and Qiang Liu.

It uses rectified flow concepts to generate a velocity vector that converts from one image domain to another.

It is able to work with unpaired images by using a feature mapping of an image. To quote:

> Let `h(x)` be a feature mapping of image `x` representing the styles that we are interested in transferring.
> Let `X_t = t*X_1 + (1-t)*X_0`. Then `H_t = h(X_t)` follows an ODE of `dH_t = del(h(X_t))^T*(X_1 - X_0)*dt`.
> Hence, to ensure that the style is transferred correctly, we propose to learn `v` such that `H_t' = h(Z_t)` with `dZ_t = v(Z_t, t)*dt` approximates `H_t` as much as possible.
> Because `dH_t' = del(h(Z_t))^T*v(Z_t, t)*dt`, we propose to minimize the following loss:
>
> `min(v, integral(t=0..1, E[||del(h(X_t))^T*(X_1 - x_0 - v(X_t, t)||^2]dt)`
>
> In practice, we set `h(x)` to be the latent representation of a classifier
> trained to distinguish the images from the two domains `pi_0` and `pi_1`,
> fine-tuned from a pre-trained ImageNet model.
> Intuitively, `del_x(h(x))` serves as a saliency score
> and re-weights coordinates so that the loss focuses on penalizing
> the error that causes significant changes on `h`.

## Environment

- Do **not** create virtual environments.
- Ask before installing any new dependencies.
- Python is version 3.10.12.
- You must use the command `python3` to run the code, not `python`.
- All dependencies should be installed globally using `pip3`.
- PyTorch version is 2.6.0+cu124

## Implementation Details

- Uses PyTorch for all training
- Uses pre-trained models from torchvision for feature extraction
- Uses a UNet architecture for the velocity field
- Images are normalized to [-1, 1] colorspace throughout the app

### Generative Mode

The model can operate in two modes:

1. **Image Translation Mode** (default): Converts images from domain 0 to domain 1 using unpaired data.
   - Example: `--domain0 data/horses --domain1 data/zebras`

2. **Generative Mode**: Generates images from random noise (standard Gaussian) to domain 1.
   - Use the magic path `"random"` for domain 0: `--domain0 random --domain1 data/faces`
   - Domain 0 will be sampled from standard normal distribution N(0,1) (unclipped)
   - This matches standard Rectified Flow generative modeling
   - Structure pairing is not applicable in this mode

## Training Procedure

Training happens in two phases:

### Phase 1: Saliency Network Training

The saliency network `h(x)` is a domain classifier (e.g., ResNet-based) that learns to distinguish images from domain 0 vs domain 1. Its latent features and gradients are used to weight the velocity loss.

**Blend Training** (optional): Set `saliency_blend_fraction` (0.0 to 1.0) to expose the classifier to interpolated images `x_t = t*x1 + (1-t)*x0` during training. This uses soft cross-entropy with targets `[1-t, t]`, which degenerates to hard labels at t=0 or t=1. The goal is to improve Jacobian quality at intermediate timesteps during velocity training.

**Label Smoothing** (optional): Set `saliency_label_smoothing` (0.0 to 1.0, typically 0.1) to prevent classifier overconfidence. Soft targets are smoothed: `targets = targets * (1 - ε) + ε / num_classes`.

**Data Augmentation** (optional): Set `saliency_augmentations` to a list of augmentation names to improve classifier robustness. Supported augmentations:
- `"color_jitter"`: Random brightness/contrast/saturation/hue. Prevents color histogram shortcuts.
- `"grayscale"`: Random grayscale conversion (p=0.1). Forces learning texture/shape.
- `"random_erasing"`: Random rectangular patch erasure. Prevents relying on single regions.
- `"gaussian_blur"`: Random Gaussian blur. Forces learning coarser structure.

This phase has two sub-phases for pretrained backbones:

1. **Backbone Warmup** (frozen backbone): Train only the head layers (`latent_proj`, `classifier`) until accuracy reaches `saliency_warmup_threshold`. This prevents random gradients from corrupting pretrained weights.

2. **Full Fine-tuning** (unfrozen backbone): Continue training the entire network until accuracy reaches `saliency_accuracy_threshold`.

Training state is saved in `runs/<run_id>/saliency_state.json`:
```json
{
  "accuracy": 95,
  "backbone_warmed_up": true
}
```

### Phase 2: Velocity Network Training

Once the saliency network is trained, it is frozen and the velocity network is trained using the saliency-weighted flow matching loss:

`loss = ||J_h(x_t) @ (v_target - v_pred)||²`

where `J_h` is the Jacobian of the saliency network's `get_latent()` function, computed via JVP (Jacobian-vector product).

**Structure-Aware Pairing** (optional): Set `structure_pairing: true` to enable intelligent image pairing during velocity training. Instead of random cross-domain pairing, this uses DINOv2 embeddings to find structurally similar images across domains. This helps when domains have diverse compositions (close-ups vs wide shots, different scenes) by matching images with similar layout/pose.

- Embeddings are cached globally in `~/.cache/rfpix2pix/dino_embeddings.db` (SQLite)
- `structure_candidates` controls how many similar images to consider (default: 8)
- Higher values = more variety, lower values = stricter matching