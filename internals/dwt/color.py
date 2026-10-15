"""Color handling: RGB <-> YCbCr (BT.601) conversion and 4:2:0 chroma resampling."""
import numpy as np

from .shapes import crop_or_pad


def split_channels(rgb: np.ndarray, color_space: str) -> dict[str, np.ndarray]:
    """
    Split an RGB array into the planes that get transformed.

    Args:
        rgb: Float array of shape (H, W, 3).
        color_space: `rgb` or `ycbcr420`.

    Returns:
        `R/G/B` planes, or `Y` (full size) plus `Cb/Cr` (half size).

    Raises:
        ValueError: If `color_space` is unsupported.
    """
    if color_space == "rgb":
        return {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}
    if color_space != "ycbcr420":
        raise ValueError(f"Unsupported color space: {color_space}")

    y, cb, cr = rgb_to_ycbcr(rgb)
    return {"Y": y, "Cb": downsample_420(cb), "Cr": downsample_420(cr)}


def merge_channels(
    channels: dict[str, np.ndarray], color_space: str, orig_shape: tuple[int, int]
) -> np.ndarray:
    """
    Inverse of `split_channels`: rebuild a float RGB array.

    Args:
        channels: Reconstructed planes keyed by channel name.
        color_space: `rgb` or `ycbcr420`.
        orig_shape: Target (height, width) for all planes.

    Returns:
        Float array of shape (H, W, 3), not yet clipped to 0-255.
    """
    height, width = orig_shape
    if color_space == "rgb":
        stacked = np.stack(
            [
                crop_or_pad(channels["R"], height, width),
                crop_or_pad(channels["G"], height, width),
                crop_or_pad(channels["B"], height, width),
            ],
            axis=-1,
        )
        return stacked

    y = crop_or_pad(channels["Y"], height, width)
    cb = crop_or_pad(upsample_420(channels["Cb"], height, width), height, width)
    cr = crop_or_pad(upsample_420(channels["Cr"], height, width), height, width)
    return ycbcr_to_rgb(y, cb, cr)


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB to full-range YCbCr (JPEG/BT.601 coefficients).

    Args:
        rgb: Float array of shape (H, W, 3).

    Returns:
        Separate Y, Cb, Cr planes, each (H, W).
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b  # brightness
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b  # blue-difference
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b  # red-difference
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """
    Convert full-range YCbCr planes back to RGB.

    Args:
        y: Luma plane.
        cb: Blue-difference chroma plane (same shape as `y`).
        cr: Red-difference chroma plane (same shape as `y`).

    Returns:
        Float array of shape (H, W, 3).
    """
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


def downsample_420(channel: np.ndarray) -> np.ndarray:
    """
    Halve a plane in both axes by averaging its 2x2 blocks.

    Args:
        channel: 2D plane; an odd last row/column is dropped.

    Returns:
        Plane of shape (H // 2, W // 2).
    """
    height, width = channel.shape
    even_h, even_w = height - height % 2, width - width % 2
    if even_h == 0 or even_w == 0:
        return channel
    cropped = channel[:even_h, :even_w]
    return cropped.reshape(even_h // 2, 2, even_w // 2, 2).mean(
        axis=(1, 3)
    )  # now 1px -> avg(2x2)


def upsample_420(channel: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Double a plane by pixel repetition, then crop/pad to the target size.

    Args:
        channel: Half-resolution plane.
        height: Target rows.
        width: Target columns.

    Returns:
        Plane of shape (height, width).
    """
    upsampled = np.repeat(
        np.repeat(channel, 2, axis=0), 2, axis=1
    )  # not an exact original, but approximation
    return crop_or_pad(upsampled, height, width)
