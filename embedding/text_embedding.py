"""Defines interface and implementation for generating text embeddings."""

from typing import Optional
from typing_extensions import override
from abc import ABC, abstractmethod

import pydantic
from sentence_transformers import SentenceTransformer
import numpy as np

from internals.utils import validate_dtypes
from exceptions.embedding import TextEmbeddingError


class TextEmbeddingI(ABC):
    """Interface class for text embedding."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerTextEmbedding(pydantic.BaseModel, TextEmbeddingI):
    """
    Text Embedding with `SentenceTransformer`.

    Attributes:
        pretrained_model_name_or_path: If it is a filepath, the model will be loaded from it.
                            If it is not a path, it first tries to download a pre-trained `SentenceTransformer` model.
                            If that fails, tries to construct a model from the Hugging Face Hub.
        model: `SentenceTransformer` model that can be used to map text to embeddings.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    pretrained_model_name_or_path: str = pydantic.Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    model: Optional[SentenceTransformer] = pydantic.Field(default=None)

    def model_post_init(self, context):
        if not self.model:
            try:
                self.model = SentenceTransformer(self.pretrained_model_name_or_path, device="cuda:0")
            except Exception as e:
                raise TextEmbeddingError(
                    f"Error while initializing embedding model: {e}"
                )

    @override
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Computes sentence embeddings.

        Args:
            texts: The sentences to embed.

        Raises:
            TypeError: If texts is not a list or not all elements in texts are strings.
            TextEmbeddingError: If embedding fails.

        Returns:
            2d-array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, the output is a 1d-array with shape [output_dimension].
        """

        validate_dtypes([texts], ["texts"], [list])
        for text in texts:
            validate_dtypes(
                [text],
                ["text"],
                [str],
            )

        try:
            embeddings = np.asarray(self.model.encode(texts), dtype=np.float32)
            return embeddings
        except Exception as e:
            raise TextEmbeddingError(f"Error while encoding texts: {e}")
