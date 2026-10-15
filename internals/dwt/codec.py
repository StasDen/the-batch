"""`.dwtz` bitstream: header + zlib payload, with encode/decode and file save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import json
import zlib

import numpy as np
from PIL import Image

from .quantization import quantize_result, dequantize_result
from .shapes import parse_channel_shapes
from .transform import dwt, idwt
from .packing import (
    flatten_quantized,
    pack_coeff_array,
    unflatten_quantized,
    unpack_coeff_array,
)
from .config import (
    DEFAULT_WAVELET,
    DEFAULT_LEVEL,
    DEFAULT_Y_STEP,
    DEFAULT_CHROMA_STEP,
    HEADER_STRUCT,
    MAGIC,
    VERSION,
    ZLIB_LEVEL,
    DEFAULT_COLOR_SPACE,
    SUPPORTED_VERSIONS
)


def encode_image_dwt(
    image: Image.Image,
    wavelet: str = DEFAULT_WAVELET,
    level: Optional[int] = DEFAULT_LEVEL,
    y_step: float = DEFAULT_Y_STEP,
    chroma_step: float = DEFAULT_CHROMA_STEP,
    color_space: str = DEFAULT_COLOR_SPACE,
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
    quantized, steps = quantize_result(result, y_step=y_step, chroma_step=chroma_step)
    arrays = flatten_quantized(quantized)
    packed_chunks: list[bytes] = []
    array_specs: list[dict] = []
    for arr in arrays:
        chunk, spec = pack_coeff_array(arr)
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
    payload = zlib.compress(b"".join(packed_chunks), ZLIB_LEVEL)
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
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported DWT bitstream version: {version}")

    header_end = header_size + header_len
    header = json.loads(data[header_size:header_end].decode("utf-8"))
    payload = zlib.decompress(data[header_end:])

    arrays: list[np.ndarray] = []
    offset = 0
    for spec in header["arrays"]:
        dtype = np.dtype(spec["dtype"])
        nbytes = spec.get("nbytes", int(np.prod(spec["shape"])) * dtype.itemsize)
        arrays.append(unpack_coeff_array(spec, payload[offset : offset + nbytes]))
        offset += nbytes

    quantized = unflatten_quantized(arrays, header["channels"], header["level"])
    orig_shape = tuple(header["orig_shape"])
    color_space = header["color_space"]
    channel_shapes = parse_channel_shapes(
        header.get("channel_shapes"),
        orig_shape=orig_shape,
        color_space=color_space,
        channels=header["channels"],
    )
    result = dequantize_result(
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
