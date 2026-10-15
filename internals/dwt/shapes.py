"""Plane-shape helpers: cropping/padding, dyadic sizes and per-channel shape."""
from typing import Optional

import numpy as np


def crop_or_pad(
    channel: np.ndarray, height: int, width: int, mode: str = "constant"
) -> np.ndarray:
    """
    Crop or pad a 2D plane to exactly (height, width).

    Args:
        channel: 2D float plane.
        height: Target rows.
        width: Target columns.
        mode: `np.pad` mode for growing the plane ("constant" = zeros,
            "edge" = replicate border pixels).

    Returns:
        Plane of shape (height, width).
    """

    cropped = channel[:height, :width]
    pad_rows = height - cropped.shape[0]
    pad_cols = width - cropped.shape[1]
    if pad_rows == 0 and pad_cols == 0:
        return cropped
    return np.pad(cropped, ((0, pad_rows), (0, pad_cols)), mode=mode)


def dyadic_shape(shape: tuple[int, int], level: int) -> tuple[int, int]:
    """
    Round a plane shape up to the next multiple of `2**level`.

    Every decomposition level halves the size, so each must stay even.
    """

    block = 2**level
    return tuple(-(-size // block) * block for size in shape)  # ceil to multiple


def chroma_shape(orig_shape: tuple[int, int]) -> tuple[int, int]:
    """
    Shape of a 4:2:0 chroma plane for an image of `orig_shape`.

    Odd rows/columns are dropped before halving, matching `downsample_420`.

    Args:
        orig_shape: Full-resolution (height, width).

    Returns:
        Chroma (height, width); `orig_shape` unchanged if either side is < 2.
    """
    height, width = orig_shape
    even_h, even_w = height - height % 2, width - width % 2
    if even_h == 0 or even_w == 0:
        return orig_shape
    return (even_h // 2, even_w // 2)


def derive_channel_shapes(
    orig_shape: tuple[int, int], color_space: str, channels: list[str]
) -> dict[str, tuple[int, int]]:
    """
    Rebuild per-channel plane shapes when the header doesn't store them.

    Args:
        orig_shape: Full-resolution (height, width).
        color_space: `rgb` or `ycbcr420`.
        channels: Channel names in stored order.

    Returns:
        Mapping of channel name to (height, width).
    """
    height, width = orig_shape
    if color_space == "rgb":
        return {name: (height, width) for name in channels}
    chroma = chroma_shape(orig_shape)
    return {name: (height, width) if name == "Y" else chroma for name in channels}


def parse_channel_shapes(
    raw_shapes: Optional[dict],
    orig_shape: tuple[int, int],
    color_space: str,
    channels: list[str],
) -> dict[str, tuple[int, int]]:
    """
    Read channel shapes from the header, falling back to derived ones.

    Args:
        raw_shapes: `channel_shapes` header field (JSON lists), or None.
        orig_shape: Full-resolution (height, width).
        color_space: `rgb` or `ycbcr420`.
        channels: Channel names in stored order.

    Returns:
        Mapping of channel name to (height, width) tuples.
    """
    if not raw_shapes:
        return derive_channel_shapes(orig_shape, color_space, channels)
    return {name: (int(shape[0]), int(shape[1])) for name, shape in raw_shapes.items()}
