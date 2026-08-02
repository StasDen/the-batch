"""Defines an interface protocol for custom vector stores supporting document addition, similarity search, etc."""

from typing import runtime_checkable, Protocol

import numpy as np

from schema.documents import BaseDocument


@runtime_checkable
class VectorStoreI(Protocol):
    """Interface class for custom VectorStore."""

    def __init__(self, persist_dir: str): ...

    def search(self, query: str, k: int, *args, **kwargs) -> list[BaseDocument]: ...

    def add_documents(
        self, docs: list[BaseDocument], embeddings: np.ndarray
    ) -> None: ...

    def load(self, persist_dir: str) -> None: ...

    def clear(self) -> None: ...
