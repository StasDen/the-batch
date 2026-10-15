"""
Wavelet image codec: DWT, dead-zone quantization, sparse packing, zlib.

Public API: `dwt`/`idwt` for the raw transform, and
`encode_image_dwt`/`decode_image_dwt`/`save_dwtz`/`load_dwtz` for `.dwtz` files.
"""

from .features import save_signatures, load_signatures
from .codec import decode_image_dwt, encode_image_dwt, load_dwtz, save_dwtz
from .transform import DWTResult, dwt, idwt

__all__ = [
    "DWTResult", "dwt", "idwt",
    "encode_image_dwt", "decode_image_dwt", "save_dwtz", "load_dwtz",
    "save_signatures", "load_signatures"
]
