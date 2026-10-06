"""Utility functions."""

import hashlib
from pathlib import Path
from typing import Any, Optional
import json
import uuid

from schema.documents import ImageDocument
from internals.dwt import load_dwtz, save_dwtz


def validate_dtypes(inputs: list, input_names: list, required_dtypes: list) -> None:
    """
    Validate if inputs corresponds to required data types.

    Args:
        inputs: List of object for type validations.
        input_names: List of object names to use in the error message.
        required_dtypes: List of expected data types.

    Raises:
        TypeError: If input does not match expected type.
    """

    for input, input_name, required_dtype in zip(inputs, input_names, required_dtypes):
        if not isinstance(input, required_dtype):
            raise TypeError(
                f"Input '{input_name}' must be of type '{required_dtype}'. Type got instead: '{type(input)}'"
            )


def save_img_docs(
    img_doc_mappings: dict[str, Any], store_dir: str, indent: int = 4
) -> None:
    """
    Saves `ImageDocument` objects as DWT-compressed files plus metadata JSON.

    Each image is encoded with discrete wavelet transform, quantization and
    zlib entropy coding into `{id}.dwtz`. Document fields (without image bytes)
    are written to `metadata.json`.

    Args:
        img_doc_mappings: Mapping of unique string IDs to `ImageDocument` instances.
        store_dir: Directory where encoded images and metadata will be written.
        indent: Indentation used when writing `metadata.json`.

    Raises:
        IOError: If the directory or files cannot be written.
    """

    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}
    for doc_id, img_doc in img_doc_mappings.items():
        if img_doc.image is not None:
            save_dwtz(store_path / f"{doc_id}.dwtz", img_doc.image)
        metadata[doc_id] = {
            "id": img_doc.id,
            "content": img_doc.content,
            "image_url": img_doc.image_url,
            "source_url": img_doc.source_url,
        }

    with open(store_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=indent)


def load_img_docs(store_dir: str) -> dict[str, ImageDocument]:
    """
    Loads `ImageDocument` objects from a DWT image store directory.

    Reads `metadata.json` and reconstructs each `{id}.dwtz` payload into a
    `PIL.Image` via inverse DWT.

    Args:
        store_dir: Directory containing `metadata.json` and `.dwtz` files.

    Returns:
        Mapping of image document IDs to deserialized `ImageDocument` instances.

    Raises:
        IOError: If the directory or files cannot be read.
        json.JSONDecodeError: If `metadata.json` is not valid JSON.
        ValueError: If a `.dwtz` bitstream is invalid.
    """

    store_path = Path(store_dir)
    with open(store_path / "metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: dict[str, ImageDocument] = {}
    for doc_id, fields in data.items():
        payload_path = store_path / f"{doc_id}.dwtz"
        image = load_dwtz(payload_path) if payload_path.exists() else None
        documents[doc_id] = ImageDocument(**fields, image=image)
    return documents


def generate_unique_doc_id(content: str, metadata: Optional[dict] = None) -> str:
    """
    Generate a unique document identifier based on content, metadata and some random salt.

    Args:
        content: The main content of the document.
        metadata: Additional metadata to incorporate into the hash.

    Returns:
        Unique document identifier in the format: hash-uuid.
    """

    combined = content + str(sorted(metadata.items()) if metadata else "")
    hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    salt = str(uuid.uuid4())
    return f"{hash}-{salt}"
