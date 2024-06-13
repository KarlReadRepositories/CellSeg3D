import torch
FIXED_NORMALIZATION = True


def normalize_inplace(
    image,
    im_max,
    im_min,
    new_max=100,
    new_min=0,
    batch_size=16384,
):
    """Modified from CellSeg3D/napari_cellseg3d/utils.py for low mem inference; in place normalize
    Normalizes a numpy array or Tensor using the max and min value."""
    im_view = image.view(-1)
    for i in range(0, len(im_view), batch_size):
        iend = i + batch_size
        im_view[i:iend] = (im_view[i:iend] - im_min) / (im_max - im_min)
        im_view[i:iend] = im_view[i:iend] * (new_max - new_min) + new_min


def normalize_inplace_global(image):
    if FIXED_NORMALIZATION:
        im_min, im_max = 0., 1.
    else:
        im_min, im_max = image.min(), image.max()
    normalize_inplace(image, im_min=im_min, im_max=im_max)

