"""Defines interface and implementation for generating image embeddings."""

from typing import Optional
from typing_extensions import override
from abc import abstractmethod, ABC

import pydantic
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor

from internals.utils import validate_dtypes
from exceptions.embedding import ImageEmbeddingError


class ImageEmbeddingI(ABC):
    """Interface class for image embedding."""

    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


class CLIPImageEmbedding(pydantic.BaseModel, ImageEmbeddingI):
    """
    Image embedding using CLIP from Hugging Face.

    Attributes:
        pretrained_model_name_or_path: Hugging Face Hub model ID or path to local model.
        model: The CLIP model instance.
        processor: CLIP processor for preparing image inputs.

    Raises:
        ImageEmbeddingError: If any exception happens during model or processor loading.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    pretrained_model_name_or_path: str = pydantic.Field(
        default="openai/clip-vit-base-patch32"
    )
    model: Optional[CLIPModel] = pydantic.Field(default=None)
    processor: Optional[CLIPProcessor] = pydantic.Field(default=None)

    def model_post_init(self, context):
        if not self.model:
            try:
                self.model = CLIPModel.from_pretrained(
                    self.pretrained_model_name_or_path,
                    device_map="cuda:0",
                    max_memory={0: 8192},
                )
            except Exception as e:
                raise ImageEmbeddingError(
                    f"Error occurred while initializing model: {e}"
                )
        if not self.processor:
            try:
                self.processor = CLIPProcessor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    use_fast=True,
                    device_map="cuda:0",  # install cuda first or use cpu
                )
            except Exception as e:
                raise ImageEmbeddingError(f"Error while initializing processor: {e}")

    @override
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """
        Computes image embeddings.

        Args:
            images: List of images to embed.

        Raises:
            TypeError: If images is not a list or contains not `Image.Image` object.
            ImageEmbeddingError: If the embedding fails.

        Returns:
            The image embeddings obtained by applying the projection layer to the pooled output of `CLIPModel`.
        """

        validate_dtypes([images], ["images"], [list])
        for image in images:
            validate_dtypes(
                [image],
                ["image"],
                [Image.Image],
            )

        try:
            tensors = self.processor(images=images, return_tensors="pt")
            embeddings = self.model.get_image_features(**tensors)
            np_embeddings = embeddings.detach().cuda().numpy().astype(np.float32)
            return np_embeddings
        except Exception as e:
            raise ImageEmbeddingError(f"Error occurred while encoding image: {e}")
