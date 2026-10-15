"""Byte-level (de)serialization of quantized subbands, dense or sparse (gap-coded)."""
import numpy as np


def flatten_quantized(quantized: dict[str, list]) -> list[np.ndarray]:
    """
    Concatenate all channels' subbands into one list, in channel order.

    Args:
        quantized: Per-channel subband lists.

    Returns:
        Flat list matching the order of `header["arrays"]`.
    """
    arrays: list[np.ndarray] = []
    for channel_arrays in quantized.values():
        arrays.extend(channel_arrays)
    return arrays


def unflatten_quantized(
    arrays: list[np.ndarray], channels: list[str], level: dict[str, int]
) -> dict[str, list]:
    """
    Inverse of `flatten_quantized`.

    Args:
        arrays: Flat subband list from the bitstream.
        channels: Channel names in stored order.
        level: Per-channel depth; each channel owns `1 + 3 * level` arrays.

    Returns:
        Per-channel subband lists.
    """
    quantized: dict[str, list] = {}
    cursor = 0
    for name in channels:
        n_arrays = 1 + 3 * int(level[name])  # 1 approx + 3 detail
        quantized[name] = arrays[cursor : cursor + n_arrays]
        cursor += n_arrays
    return quantized


def pack_coeff_array(arr: np.ndarray) -> tuple[bytes, dict]:
    """
    Serialize one quantized subband, sparse or dense, whichever is smaller.

    Sparse layout: gap-encoded nonzero positions, then the nonzero values.

    Args:
        arr: Quantized integer subband.

    Returns:
        Raw bytes and a spec (`shape`, `dtype`, `encoding`, `nbytes`, plus
        `nnz`/`idx_dtype` for sparse) to store in the header.
    """
    arr = np.ascontiguousarray(arr)  # ensure a single predictable copy
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


def unpack_coeff_array(spec: dict, buf: bytes) -> np.ndarray:
    """
    Inverse of `pack_coeff_array`.

    Args:
        spec: Header entry for this array; a missing `encoding` means
            dense (v1 files).
        buf: This array's bytes only.

    Returns:
        Integer array of `spec["shape"]`.
    """
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
