"""Static settings for the DWT codec."""

from __future__ import annotations

import struct
from pathlib import Path

# --- Bitstream format ---
MAGIC = b"DWTZ"
VERSION = 2  # 2 = per-array `encoding`/`nbytes` in header; v1 still decodes
SUPPORTED_VERSIONS = (1, 2)
HEADER_STRUCT = struct.Struct(">4sBI")  # magic, version, header length
ZLIB_LEVEL = 9

# --- Transform ---
DEFAULT_WAVELET = "haar"
DEFAULT_LEVEL = 4
DEFAULT_COLOR_SPACE = "ycbcr420"
DWT_MODE = "periodization"  # exact coeff count, no filter overhang

# --- Quantization ---
DEFAULT_Y_STEP = 16.0
DEFAULT_CHROMA_STEP = 24.0
APPROX_STEP_SCALE = 0.75     # approximation band: base * 0.75
DETAIL_STEP_GROWTH = 3.0     # finest detail band: base * (1 + 3.0) = 4x
MIN_APPROX_STEP = 1.0
LUMA_CHANNELS = frozenset({"Y", "R", "G", "B"})  # use y_step; others use chroma_step

BASE_DIR = Path(__file__).resolve().parent
WAVELET_SIGNATURES_PATH = BASE_DIR / "wvlt_signatures.npz"
SIGNATURE_SIZE = 256  # thumbnail side; every image is compared at this size
