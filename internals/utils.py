"""Utility functions."""

import hashlib
from typing import Any, Optional
import json
import uuid
import io
import base64

from PIL import Image

from schema.documents import ImageDocument


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


def img_doc_to_serializable_dict(img_doc: ImageDocument) -> dict:
    """
    Converts an `ImageDocument` instance into a serializable dictionary.

    This function transforms the given `ImageDocument` object into a dictionary
    that can be easily serialized (e.g., to JSON). Specifically:
    - Metadata fields are preserved as key-value pairs.
    - If the 'image' field contains a `PIL.Image.Image` object, it is converted into a
      base64-encoded PNG format string (img_base64) to ensure compatibility
      with text-based serialization formats.

    Args:
        img_doc: The `ImageDocument` instance to convert.

    Returns:
        - All metadata fields from the ImageDocument (id, type, content,
            source_url, image_url).
        - An additional key 'img_base64' with the base64-encoded image string
            if the image field is present; otherwise, `None`.
    """

    img_base64 = None
    if img_doc.image:
        buffered = io.BytesIO()
        img_doc.image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {**img_doc.metadata, "img_base64": img_base64}


def save_img_docs_to_json(
    img_doc_mappings: dict[str, Any], img_doc_store_path: str, indent: int = 4
) -> None:
    """
    Serializes and saves a collection of `ImageDocument` objects to a JSON file.

    This function transforms a dictionary mapping of unique image document IDs to
    `ImageDocument` instances into a serializable dictionary format. Each `ImageDocument`
    is converted using the `img_doc_to_serializable_dict` function to ensure
    that complex types such as `PIL.Image.Image` are converted to base64-encoded strings.
    The resulting data is then written to a JSON file at the specified path.

    Args:
        img_doc_mappings: A dictionary mapping of unique string IDs to `ImageDocument` instances.
        img_doc_store_path: The file path where the serialized JSON data will be saved.

    Raises:
        IOError: If the file cannot be opened or written to.
    """

    serializable_data = {
        id: img_doc_to_serializable_dict(img) for id, img in img_doc_mappings.items()
    }

    with open(img_doc_store_path, "w") as f:
        json.dump(serializable_data, f, indent=indent)


def load_img_docs_from_json(img_doc_store_path: str) -> dict[str, ImageDocument]:
    """
    Loads and deserializes `ImageDocument` objects from a JSON file.

    This function reads JSON data from the specified file path, where the data
    represents a dictionary mapping of unique image document IDs to their serialized
    dictionary forms. Each serialized dictionary is converted back into an
    ImageDocument instance using the `from_serializable_to_img_doc` function,
    which handles decoding of any embedded base64-encoded images.

    Args:
        img_doc_store_path: The file path from which to load the serialized JSON.

    Returns:
        A dictionary mapping of image document IDs to their deserialized `ImageDocument` instances.

    Raises:
        IOError: If the file cannot be opened or read.
        json.JSONDecodeError: If the file content is not valid JSON.
    """

    with open(img_doc_store_path, "r") as f:
        data = json.load(f)

    return {
        id: from_serializable_to_img_doc(serializable_content)
        for id, serializable_content in data.items()
    }


def from_serializable_to_img_doc(data: dict) -> ImageDocument:
    """
    Reconstructs an `ImageDocument` instance from a serializable dictionary.

    This function reverses the transformation done by `img_doc_to_serializable_dict`.
    It reconstructs an `ImageDocument` object by:
    - Extracting metadata fields from the provided dictionary.
    - Decoding 'img_base64' field (if present) from a base64-encoded
      PNG string into a `PIL.Image.Image` object and assigning it to the 'image' field.

    Args:
        data:
            - Metadata fields ('id', 'type', 'content', 'source_url', 'image_url').
            - Optionally, 'image_base64' (a base64-encoded PNG image).

    Returns:
        The reconstructed `ImageDocument` object with metadata fields and the decoded image.
    """

    image_data = data.pop("img_base64", None)
    image_obj = None
    if image_data:
        img_bytes = base64.b64decode(image_data)
        image_obj = Image.open(io.BytesIO(img_bytes))
    return ImageDocument(**data, image=image_obj)


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
