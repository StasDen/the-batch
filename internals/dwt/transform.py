"""Multilevel 2D wavelet transform (`dwt`/`idwt`) and coefficient list (un)flattening."""
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import numpy as np
import pywt

from .config import DEFAULT_WAVELET, DWT_MODE, DEFAULT_LEVEL, DEFAULT_COLOR_SPACE
from .shapes import crop_or_pad, dyadic_shape
from .color import split_channels, merge_channels


@dataclass
class DWTResult:
    """Unquantized wavelet coefficients plus invert metadata."""

    coeffs: dict[str, list]
    orig_shape: tuple[int, int]
    wavelet: str
    level: dict[str, int]
    color_space: str
    channel_shapes: dict[str, tuple[int, int]]


def dwt(
    image: Image.Image,
    wavelet: str = DEFAULT_WAVELET,
    level: Optional[int] = None,
    color_space: str = DEFAULT_COLOR_SPACE,
) -> DWTResult:
    """
    Decompose an image with a 2D multilevel discrete wavelet transform.

    RGB images are converted to YCbCr with 4:2:0 chroma (default) so the two
    chroma channels are transformed at half resolution. Each channel is
    decomposed independently with `pywt.wavedec2`.

    Args:
        image: Source image. Converted to RGB if needed.
        wavelet: PyWavelets wavelet name. Defaults to `haar`.
        level: Decomposition depth. If None, uses up to 4, capped by
            `pywt.dwt_max_level` per channel.
        color_space: `ycbcr420` (default) or `rgb`.

    Returns:
        `DWTResult` with per-channel coefficient lists and invert metadata.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    channels = split_channels(rgb, color_space)

    coeffs: dict[str, list] = {}
    levels: dict[str, int] = {}
    channel_shapes: dict[str, tuple[int, int]] = {}
    for name, channel in channels.items():
        channel_level = resolve_level(channel.shape, wavelet, level)
        channel_shapes[name] = (int(channel.shape[0]), int(channel.shape[1]))
        target_height, target_width = dyadic_shape(channel.shape, channel_level)
        # "edge" avoids a hard black border that would create fake high-frequency detail
        padded = crop_or_pad(channel, target_height, target_width, mode="edge")
        coeffs[name] = pywt.wavedec2(
            padded, wavelet=wavelet, level=channel_level, mode=DWT_MODE
        )
        levels[name] = channel_level

    return DWTResult(
        coeffs=coeffs,
        orig_shape=(height, width),
        wavelet=wavelet,
        level=levels,
        color_space=color_space,
        channel_shapes=channel_shapes,
    )


def idwt(result: DWTResult) -> Image.Image:
    """
    Invert a `dwt` result back to an RGB `PIL.Image`.

    Args:
        result: Coefficient bundle produced by `dwt` (or dequantization).

    Returns:
        Reconstructed RGB image cropped to the original spatial size.
    """

    reconstructed: dict[str, np.ndarray] = {}
    for name, coeffs in result.coeffs.items():
        channel = pywt.waverec2(coeffs, wavelet=result.wavelet, mode=DWT_MODE)
        channel = np.asarray(channel, dtype=np.float32)
        shape = result.channel_shapes.get(name)
        if shape is not None:
            channel = crop_or_pad(channel, shape[0], shape[1])
        reconstructed[name] = channel

    rgb = merge_channels(reconstructed, result.color_space, result.orig_shape)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resolve_level(shape: tuple[int, ...], wavelet: str, level: Optional[int]) -> int:
    """
    Clamp the requested decomposition depth to what the plane supports.

    Args:
        shape: Plane shape; only the first two dims are used.
        wavelet: PyWavelets wavelet name.
        level: Requested depth, or None for `DEFAULT_LEVEL`.

    Returns:
        Depth in [1, `pywt.dwt_max_level`].
    """
    max_level = pywt.dwt_max_level(min(shape[:2]), pywt.Wavelet(wavelet).dec_len)
    max_level = max(1, max_level)
    if level is None:
        return min(DEFAULT_LEVEL, max_level)
    return max(1, min(level, max_level))


def flatten_coeffs(coeffs: list) -> list[np.ndarray]:
    """
    Flatten `pywt.wavedec2` output into one list of subbands.

    Args:
        coeffs: `[cA, (cH, cV, cD), ...]` coarsest to finest.

    Returns:
        `[cA, cH1, cV1, cD1, cH2, ...]`.
    """
    arrays = [np.asarray(coeffs[0])]
    for details in coeffs[1:]:
        arrays.extend(np.asarray(band) for band in details)
    return arrays


def unflatten_coeffs(arrays: list[np.ndarray], level: int) -> list:
    """
    Inverse of `flatten_coeffs`, for `pywt.waverec2`.

    Args:
        arrays: Flat subband list of length `1 + 3 * level`.
        level: Decomposition depth.

    Returns:
        `[cA, (cH, cV, cD), ...]`.
    """
    details = []
    cursor = 1
    for _ in range(level):
        details.append((arrays[cursor], arrays[cursor + 1], arrays[cursor + 2]))
        cursor += 3
    return [arrays[0], *details]
