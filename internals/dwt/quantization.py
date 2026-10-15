"""Dead-zone quantization of wavelet subbands, with coarser steps for finer detail bands."""
import numpy as np

from .transform import flatten_coeffs, unflatten_coeffs, DWTResult
from .config import LUMA_CHANNELS


def subband_step(base_step: float, band_index: int, n_detail: int) -> float:
    if band_index == 0:
        return max(base_step * 0.75, 1.0)  # approx bands quantized differently
    fineness = band_index / n_detail
    return base_step * (1.0 + 3.0 * fineness)  # finer (later) detail bands get coarser quantization


def pack_int(values: np.ndarray) -> np.ndarray:
    """
    Cast integer-valued floats to the smallest signed int dtype that fits.

    Args:
        values: Rounded quantized coefficients.

    Returns:
        Array as int8, int16 or int32 (int8 if empty).
    """
    if values.size == 0:
        return values.astype(np.int8)
    peak = int(np.max(np.abs(values)))
    if peak <= np.iinfo(np.int8).max:
        return values.astype(np.int8)
    if peak <= np.iinfo(np.int16).max:
        return values.astype(np.int16)
    return values.astype(np.int32)


def quantize_array(arr: np.ndarray, step: float) -> np.ndarray:
    """
    Dead-zone quantize one subband.

    Values with |x| < step / 2 become 0, which is what makes detail
    bands sparse; the rest are rounded to the nearest multiple of `step`.

    Args:
        arr: Float coefficients.
        step: Quantizer step size.

    Returns:
        Integer array of quantization indices.
    """
    quantized = np.round(arr / step)
    quantized[np.abs(arr) < step / 2.0] = 0
    return pack_int(quantized)


def quantize_result(
    result: DWTResult, y_step: float, chroma_step: float
) -> tuple[dict[str, list], dict[str, list[float]]]:
    """
    Quantize every subband of every channel.

    Args:
        result: Unquantized coefficients from `dwt`.
        y_step: Base step for luma (and RGB) channels.
        chroma_step: Base step for chroma channels.

    Returns:
        Quantized arrays and the per-subband steps (needed to dequantize).
    """
    quantized: dict[str, list] = {}
    steps: dict[str, list[float]] = {}
    for name, coeffs in result.coeffs.items():
        base = y_step if name in LUMA_CHANNELS else chroma_step
        coeffs_flat = flatten_coeffs(coeffs)
        n_detail = max(len(coeffs_flat) - 1, 1)
        channel_steps = [
            subband_step(base, i, n_detail) for i in range(len(coeffs_flat))
        ]
        quantized[name] = [
            quantize_array(arr, step) for arr, step in zip(coeffs_flat, channel_steps)
        ]
        steps[name] = channel_steps
    return quantized, steps


def dequantize_result(
    quantized: dict[str, list],
    wavelet: str,
    orig_shape: tuple[int, int],
    color_space: str,
    level: dict[str, int],
    steps: dict[str, list[float]],
    channel_shapes: dict[str, tuple[int, int]],
) -> DWTResult:
    """
    Scale quantization indices back to float coefficients.

    Args:
        quantized: Per-channel integer subbands.lN11-p3Ce#1!

        wavelet: Wavelet name stored in the header.
        orig_shape: Original image (height, width).
        color_space: `rgb` or `ycbcr420`.
        level: Per-channel decomposition depth.
        steps: Per-channel, per-subband steps used when encoding.
        channel_shapes: Unpadded plane shapes for cropping in `idwt`.

    Returns:
        `DWTResult` ready for `idwt`.
    """
    coeffs: dict[str, list] = {}
    for name, arrays in quantized.items():
        floats = [
            arr.astype(np.float32) * np.float32(step)
            for arr, step in zip(arrays, steps[name])
        ]
        coeffs[name] = unflatten_coeffs(floats, level[name])
    return DWTResult(
        coeffs=coeffs,
        orig_shape=orig_shape,
        wavelet=wavelet,
        level=level,
        color_space=color_space,
        channel_shapes=channel_shapes,
    )
