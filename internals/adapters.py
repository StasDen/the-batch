"""Module containing adapter classes to integrate internal implementations with external interfaces."""

import pydantic
import numpy as np

from embedding.text_embedding import TextEmbeddingI
from internals.utils import validate_dtypes


class ChromaVectorStoreAdapter(pydantic.BaseModel):
    """
    Adapt `TextEmbeddingI` object to follow `langchain_core.embeddings.Embeddings` interface.

    Attributes:
        embedding_func: `TextEmbeddingI` object for text embedding.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    embedding_func: TextEmbeddingI

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed input text.

        Args:
            text: Text to embed.

        Raises:
            TypeError: If text is not a string.

        Returns:
            np.ndarray: Embedding vector.
        """

        validate_dtypes([text], ["text"], [str])
        return self.embedding_func.encode([text])[0]
