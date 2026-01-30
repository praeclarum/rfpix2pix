# RF Pix2pix

This is a neural network specializing in image-to-image translation using unpaired data.

It is based on Section 5.3 of the paper "Flow Straight and Fast: Learning to Generate Data and Transfer Data with Rectified Flow" by Xingchao Liu, Chenyue Gong, and Qiang Liu.

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

## Implementation Details

- Uses PyTorch for all training
- Uses pre-trained models from torchvision for feature extraction
- Uses a UNet architecture for the velocity field