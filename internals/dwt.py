"""Discrete wavelet transform helpers and a quantized + entropy-coded image codec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import json
import struct
import zlib

import numpy as np
import pywt
from PIL import Image

MAGIC = b"DWTZ"
VERSION = 1
DEFAULT_WAVELET = "haar"
DEFAULT_LEVEL = 4
DEFAULT_Y_STEP = 16.0
DEFAULT_CHROMA_STEP = 24.0
DWT_MODE = "periodization"
SPARSE_ZERO_RATIO = 0.5
HEADER_STRUCT = struct.Struct(">4sBI")


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
    color_space: str = "ycbcr420",
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
    channels = _split_channels(rgb, color_space)

    coeffs: dict[str, list] = {}
    levels: dict[str, int] = {}
    channel_shapes: dict[str, tuple[int, int]] = {}
    for name, channel in channels.items():
        channel_level = _resolve_level(channel.shape, wavelet, level)
        channel_shapes[name] = (int(channel.shape[0]), int(channel.shape[1]))
        target_height, target_width = _dyadic_shape(channel.shape, channel_level)
        # "edge" avoids a hard black border that would create fake high-frequency detail
        padded = _crop_or_pad(channel, target_height, target_width, mode="edge")
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
            channel = _crop_or_pad(channel, shape[0], shape[1])
        reconstructed[name] = channel

    rgb = _merge_channels(reconstructed, result.color_space, result.orig_shape)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def encode_image_dwt(
    image: Image.Image,
    wavelet: str = DEFAULT_WAVELET,
    level: Optional[int] = DEFAULT_LEVEL,
    y_step: float = DEFAULT_Y_STEP,
    chroma_step: float = DEFAULT_CHROMA_STEP,
    color_space: str = "ycbcr420",
) -> bytes:
    """
    Compress an image: DWT, dead-zone quantization, zlib entropy coding.

    Args:
        image: Source image.
        wavelet: Wavelet used for `dwt`.
        level: Decomposition depth passed to `dwt`.
        y_step: Base quantizer step for luma (or all RGB channels).
        chroma_step: Base quantizer step for chroma channels.
        color_space: Color transform used before DWT.

    Returns:
        Custom `.dwtz` bitstream (header + zlib payload).
    """

    result = dwt(image, wavelet=wavelet, level=level, color_space=color_space)
    quantized, steps = _quantize_result(result, y_step=y_step, chroma_step=chroma_step)
    arrays = _flatten_quantized(quantized)
    packed_chunks: list[bytes] = []
    array_specs: list[dict] = []
    for arr in arrays:
        chunk, spec = _pack_coeff_array(arr)
        packed_chunks.append(chunk)
        array_specs.append(spec)

    header = {
        "wavelet": result.wavelet,
        "orig_shape": list(result.orig_shape),
        "color_space": result.color_space,
        "level": result.level,
        "steps": steps,
        "channels": list(quantized.keys()),
        "channel_shapes": {
            name: [h, w] for name, (h, w) in result.channel_shapes.items()
        },
        "arrays": array_specs,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = zlib.compress(b"".join(packed_chunks), 9)
    return (
        HEADER_STRUCT.pack(MAGIC, VERSION, len(header_bytes)) + header_bytes + payload
    )


def decode_image_dwt(data: bytes) -> Image.Image:
    """
    Decode a `.dwtz` bitstream to an RGB `PIL.Image`.

    Args:
        data: Bytes produced by `encode_image_dwt`.

    Returns:
        Reconstructed image.

    Raises:
        ValueError: If the bitstream magic or version is invalid.
    """

    header_size = HEADER_STRUCT.size
    if len(data) < header_size:
        raise ValueError("Invalid DWT bitstream: truncated header")

    magic, version, header_len = HEADER_STRUCT.unpack(data[:header_size])
    if magic != MAGIC:
        raise ValueError("Invalid DWT bitstream: bad magic")
    if version != VERSION:
        raise ValueError(f"Unsupported DWT bitstream version: {version}")

    header_end = header_size + header_len
    header = json.loads(data[header_size:header_end].decode("utf-8"))
    payload = zlib.decompress(data[header_end:])

    arrays: list[np.ndarray] = []
    offset = 0
    for spec in header["arrays"]:
        dtype = np.dtype(spec["dtype"])
        nbytes = spec.get("nbytes", int(np.prod(spec["shape"])) * dtype.itemsize)
        arrays.append(_unpack_coeff_array(spec, payload[offset : offset + nbytes]))
        offset += nbytes

    quantized = _unflatten_quantized(arrays, header["channels"], header["level"])
    orig_shape = tuple(header["orig_shape"])
    color_space = header["color_space"]
    channel_shapes = _parse_channel_shapes(
        header.get("channel_shapes"),
        orig_shape=orig_shape,
        color_space=color_space,
        channels=header["channels"],
    )
    result = _dequantize_result(
        quantized,
        wavelet=header["wavelet"],
        orig_shape=orig_shape,
        color_space=color_space,
        level=header["level"],
        steps=header["steps"],
        channel_shapes=channel_shapes,
    )
    return idwt(result)


def save_dwtz(path: Union[str, Path], image: Image.Image, **encode_kwargs: Any) -> None:
    """Encode `image` and write a `.dwtz` file."""

    Path(path).write_bytes(encode_image_dwt(image, **encode_kwargs))


def load_dwtz(path: Union[str, Path]) -> Image.Image:
    """Read a `.dwtz` file and reconstruct a `PIL.Image`."""

    return decode_image_dwt(Path(path).read_bytes())


def _split_channels(rgb: np.ndarray, color_space: str) -> dict[str, np.ndarray]:
    if color_space == "rgb":
        return {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}
    if color_space != "ycbcr420":
        raise ValueError(f"Unsupported color space: {color_space}")

    y, cb, cr = _rgb_to_ycbcr(rgb)
    return {"Y": y, "Cb": _downsample_420(cb), "Cr": _downsample_420(cr)}


def _merge_channels(
    channels: dict[str, np.ndarray], color_space: str, orig_shape: tuple[int, int]
) -> np.ndarray:
    height, width = orig_shape
    if color_space == "rgb":
        stacked = np.stack(
            [
                _crop_or_pad(channels["R"], height, width),
                _crop_or_pad(channels["G"], height, width),
                _crop_or_pad(channels["B"], height, width),
            ],
            axis=-1,
        )
        return stacked

    y = _crop_or_pad(channels["Y"], height, width)
    cb = _crop_or_pad(_upsample_420(channels["Cb"], height, width), height, width)
    cr = _crop_or_pad(_upsample_420(channels["Cr"], height, width), height, width)
    return _ycbcr_to_rgb(y, cb, cr)


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


def _downsample_420(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    even_h, even_w = height - height % 2, width - width % 2
    if even_h == 0 or even_w == 0:
        return channel
    cropped = channel[:even_h, :even_w]
    return cropped.reshape(even_h // 2, 2, even_w // 2, 2).mean(axis=(1, 3))


def _upsample_420(channel: np.ndarray, height: int, width: int) -> np.ndarray:
    upsampled = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    return _crop_or_pad(upsampled, height, width)


def _chroma_shape(orig_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = orig_shape
    even_h, even_w = height - height % 2, width - width % 2
    if even_h == 0 or even_w == 0:
        return orig_shape
    return (even_h // 2, even_w // 2)


def _derive_channel_shapes(
    orig_shape: tuple[int, int], color_space: str, channels: list[str]
) -> dict[str, tuple[int, int]]:
    height, width = orig_shape
    if color_space == "rgb":
        return {name: (height, width) for name in channels}
    chroma = _chroma_shape(orig_shape)
    return {name: (height, width) if name == "Y" else chroma for name in channels}


def _parse_channel_shapes(
    raw_shapes: Optional[dict],
    orig_shape: tuple[int, int],
    color_space: str,
    channels: list[str],
) -> dict[str, tuple[int, int]]:
    if not raw_shapes:
        return _derive_channel_shapes(orig_shape, color_space, channels)
    return {name: (int(shape[0]), int(shape[1])) for name, shape in raw_shapes.items()}


def _crop_or_pad(channel: np.ndarray, height: int, width: int) -> np.ndarray:
    cropped = channel[:height, :width]
    if cropped.shape[0] == height and cropped.shape[1] == width:
        return cropped
    out = np.zeros((height, width), dtype=np.float32)
    out[: cropped.shape[0], : cropped.shape[1]] = cropped
    return out


def _resolve_level(shape: tuple[int, ...], wavelet: str, level: Optional[int]) -> int:
    max_level = pywt.dwt_max_level(min(shape[:2]), pywt.Wavelet(wavelet).dec_len)
    max_level = max(1, max_level)
    if level is None:
        return min(DEFAULT_LEVEL, max_level)
    return max(1, min(level, max_level))


def _subband_step(base_step: float, band_index: int, n_detail: int) -> float:
    if band_index == 0:
        return max(base_step * 0.75, 1.0)
    # finer (later) detail bands get coarser quantization
    fineness = band_index / n_detail
    return base_step * (1.0 + 3.0 * fineness)


def _pack_int(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.int8)
    peak = int(np.max(np.abs(values)))
    if peak <= np.iinfo(np.int8).max:
        return values.astype(np.int8)
    if peak <= np.iinfo(np.int16).max:
        return values.astype(np.int16)
    return values.astype(np.int32)


def _quantize_array(arr: np.ndarray, step: float) -> np.ndarray:
    quantized = np.round(arr / step)
    quantized[np.abs(arr) < step / 2.0] = 0
    return _pack_int(quantized)


def _quantize_result(
    result: DWTResult, y_step: float, chroma_step: float
) -> tuple[dict[str, list], dict[str, list[float]]]:
    quantized: dict[str, list] = {}
    steps: dict[str, list[float]] = {}
    for name, coeffs in result.coeffs.items():
        base = y_step if name in {"Y", "R", "G", "B"} else chroma_step
        arrays = _flatten_coeffs(coeffs)
        n_detail = max(len(arrays) - 1, 1)
        channel_steps = [_subband_step(base, i, n_detail) for i in range(len(arrays))]
        quantized[name] = [
            _quantize_array(arr, step) for arr, step in zip(arrays, channel_steps)
        ]
        steps[name] = channel_steps
    return quantized, steps


def _dequantize_result(
    quantized: dict[str, list],
    wavelet: str,
    orig_shape: tuple[int, int],
    color_space: str,
    level: dict[str, int],
    steps: dict[str, list[float]],
    channel_shapes: dict[str, tuple[int, int]],
) -> DWTResult:
    coeffs: dict[str, list] = {}
    for name, arrays in quantized.items():
        floats = [
            arr.astype(np.float32) * np.float32(step)
            for arr, step in zip(arrays, steps[name])
        ]
        coeffs[name] = _unflatten_coeffs(floats, level[name])
    return DWTResult(
        coeffs=coeffs,
        orig_shape=orig_shape,
        wavelet=wavelet,
        level=level,
        color_space=color_space,
        channel_shapes=channel_shapes,
    )


def _flatten_coeffs(coeffs: list) -> list[np.ndarray]:
    arrays = [np.asarray(coeffs[0])]
    for details in coeffs[1:]:
        arrays.extend(np.asarray(band) for band in details)
    return arrays


def _unflatten_coeffs(arrays: list[np.ndarray], level: int) -> list:
    details = []
    cursor = 1
    for _ in range(level):
        details.append((arrays[cursor], arrays[cursor + 1], arrays[cursor + 2]))
        cursor += 3
    return [arrays[0], *details]


def _flatten_quantized(quantized: dict[str, list]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for channel_arrays in quantized.values():
        arrays.extend(channel_arrays)
    return arrays


def _unflatten_quantized(
    arrays: list[np.ndarray], channels: list[str], level: dict[str, int]
) -> dict[str, list]:
    quantized: dict[str, list] = {}
    cursor = 0
    for name in channels:
        n_arrays = 1 + 3 * int(level[name])
        quantized[name] = arrays[cursor : cursor + n_arrays]
        cursor += n_arrays
    return quantized


def _pack_coeff_array(arr: np.ndarray) -> tuple[bytes, dict]:
    arr = np.ascontiguousarray(arr)  # ensure a single predictable copy to process below
    flat = arr.ravel()
    meta = {"shape": list(arr.shape), "dtype": arr.dtype.str}

    nz_idx = np.flatnonzero(flat)
    if nz_idx.size == 0:
        meta.update(
            encoding="sparse", nz_total=0, idx_dtype=np.dtype(np.uint16).str, nbytes=0
        )
        return b"", meta

    if nz_idx.size < flat.size:
        gaps = np.diff(nz_idx, prepend=0)
        idx_dtype = np.uint16 if gaps.max() <= np.iinfo(np.uint16).max else np.uint32
        gaps = gaps.astype(idx_dtype)
        vals = flat[nz_idx]
        if gaps.nbytes + vals.nbytes < flat.nbytes:
            result = (
                gaps.tobytes() + vals.tobytes()
            )  # save 2 parts: indices & actual values
            meta.update(
                encoding="sparse",
                nz_total=int(nz_idx.size),
                idx_dtype=np.dtype(idx_dtype).str,
                nbytes=len(result),
            )
            return result, meta

    meta.update(encoding="dense", nbytes=flat.nbytes)
    return flat.tobytes(), meta


def _unpack_coeff_array(spec: dict, buf: bytes) -> np.ndarray:
    dtype = np.dtype(spec["dtype"])
    shape = tuple(spec["shape"])
    if spec.get("encoding", "dense") == "dense":
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    out = np.zeros(int(np.prod(shape)), dtype=dtype)
    nz_total = spec["nz_total"]
    if nz_total:
        idx_dtype = np.dtype(spec["idx_dtype"])
        split = nz_total * idx_dtype.itemsize
        idx = np.cumsum(
            np.frombuffer(buf[:split], dtype=idx_dtype), dtype=np.int64
        )  # decode the result of np.diff()
        out[idx] = np.frombuffer(buf[split:], dtype=dtype)
    return out.reshape(shape)


def _crop_or_pad(
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


def _dyadic_shape(shape: tuple[int, int], level: int) -> tuple[int, int]:
    """
    Round a plane shape up to the next multiple of `2**level`.

    Every decomposition level halves the size, so each must stay even.
    """

    block = 2**level
    return tuple(-(-size // block) * block for size in shape)  # ceil to multiple
