"""Fixed-length wavelet signatures for image-to-image comparison."""

from functools import lru_cache

import numpy as np
from PIL import Image

from .transform import dwt
from .config import WAVELET_SIGNATURES_PATH, SIGNATURE_SIZE
from ..logger import logger


def wavelet_signature(image: Image.Image) -> np.ndarray:
    """
    Build a zero-mean, unit-length signature from a fixed-size luma thumbnail.

    Uses only the coarsest approximation band (global layout). Subtracting the
    mean makes the dot product a correlation, so unrelated images score near 0.

    Args:
        image: Source image of any size.

    Returns:
        1D float32 vector of unit length (all zeros for a perfectly flat image).
    """

    thumbnail = image.convert("L").resize((SIGNATURE_SIZE, SIGNATURE_SIZE))
    result = dwt(thumbnail.convert("RGB"), color_space="rgb")
    approximation = result.coeffs["R"][0].ravel()  # R == G == B for grayscale

    centered = approximation - approximation.mean()  # TODO: try detail coeffs
    return (centered / (np.linalg.norm(centered) + 1e-8)).astype(np.float32)


def save_signatures(img_documents: dict[str, Image.Image]) -> None:
    """
    Save DWT signatures of the images.

    Args:
        img_documents: Dict of images (id -> PIL image)
    """
    signatures = {
        id: wavelet_signature(img) for id, img in img_documents.items()
    }
    logger.info("Saving images DWT signatures...")
    np.savez(WAVELET_SIGNATURES_PATH, **signatures)


@lru_cache(maxsize=1)
def load_signatures() -> dict[str, np.ndarray]:
    """
    Load all wavelet signatures from disk.

    Returns:
        Mapping of image document ID to its unit-length signature.
    """

    with np.load(WAVELET_SIGNATURES_PATH) as data:
        return {doc_id: data[doc_id] for doc_id in data.files}
