"""Chroma-based VectorStore supporting multiple document types with embeddings, retrieval, etc."""

from typing_extensions import override
from typing import Optional, Literal, ClassVar, Callable

import pydantic
from langchain.embeddings.base import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np

from schema.documents import ImageDocument, TextDocument
from schema.documents import BaseDocument
from internals.utils import validate_dtypes
from internals.logger import logger
from internals.adapters import ChromaVectorStoreAdapter
from exceptions.vectorstore import (
    VectorStoreInitializationError,
    VectorStoreDocumentAdditionError,
    VectorStoreClearingError,
    VectorStoreLoadingError,
    VectorStoreSearchError,
)

to_base_document = Callable[[Document], BaseDocument]


class ChromaVectorStore(pydantic.BaseModel):
    """
    Vector Store implementation using Chroma, supporting multiple document types with embeddings.

    Attributes:
        collection_name: The name of the collection used in the Chroma vector store.
        embedding_func: A text embedding function or LangChain-compatible embedding object used to
                            convert text into vector representations.
        vectorstore: Chroma vector store instance.
        retriever: Retriever for similarity and MMR search.
        search_type: Type of retrieval search to use.

    Raises:
        ValidationError: If any attribute does not match expected data type.
        VectorStoreInitializationError: If vectorestore initialization fails.
    """

    _supported_documents: ClassVar = {TextDocument, ImageDocument}
    _to_base_documents: ClassVar = {
        TextDocument.type: lambda langchain_document: TextDocument(
            id=langchain_document.metadata["id"],
            content=langchain_document.metadata["content"],
            source_url=langchain_document.metadata["source_url"],
        ),
        ImageDocument.type: lambda langchain_document: ImageDocument(
            id=langchain_document.metadata["id"],
            content=langchain_document.metadata["content"],
            image_url=langchain_document.metadata["image_url"],
            source_url=langchain_document.metadata["source_url"],
        ),
    }

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    collection_name: str
    vectorstore: Optional[Chroma] = pydantic.Field(default=None, repr=False)
    persist_dir: str
    embedding_func: Optional[ChromaVectorStoreAdapter | Embeddings] = pydantic.Field(
        default=None
    )
    search_type: Literal["similarity", "mmr"] = pydantic.Field(default="similarity")
    retriever: Optional[VectorStoreRetriever] = pydantic.Field(default=None)

    def model_post_init(self, context):
        try:
            if not self.vectorstore:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_func,
                    persist_directory=self.persist_dir,
                )
        except Exception as e:
            raise VectorStoreInitializationError(
                f"Error while initializing vectorstore: {e}"
            )
        try:
            if not self.retriever:
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type
                )
        except Exception as e:
            raise VectorStoreInitializationError(
                f"Error while initializing vectorstore retriever: {e}"
            )

    @classmethod
    def add_supported_documents(
        cls, document: BaseDocument, to_base_document: to_base_document
    ) -> None:
        """
        Registers a new supported document type for a vector store.

        Args:
            document: Example instance of a supported document type.
            to_base_document: A function to convert LangChain Document into the BaseDocument type.

        Raises:
            TypeError: If document or to_base_document is not of the expected type.
            DocumentAdditionError: If document addition fails.
        """

        validate_dtypes(
            [document, to_base_document],
            ["document", "to_base_document"],
            [BaseDocument, to_base_document],
        )

        try:
            cls._supported_documents.add(document)
            cls._to_base_documents[document] = to_base_document
        except Exception as e:
            raise VectorStoreDocumentAdditionError(f"Error while adding supported document: {e}")

    @override
    def clear(self):
        """
        Clears the vector store, removing all stored documents and embeddings.

        Raises:
            VectoreStoreClearingError: If vectorestore clearing fails.
        """

        try:
            self.vectorstore._collection.delete()
        except Exception as e:
            raise VectorStoreClearingError(f"Error while clearing vectorstore: {e}")

    @override
    def add_documents(
        self, documents: list[BaseDocument], embeddings: np.ndarray
    ) -> None:
        """
        Adds documents and their embeddings to the Chroma vector store.

        Args:
            documents: List of BaseDocument instances to store.
            embeddings: 2D-array of embedding vectors corresponding to provided documents.

        Raises:
            TypeError: If embeddings are not a `numpy.ndarray`, or if a document type is not supported.
            DocumentAdditionError: If document addition fails.
        """

        validate_dtypes(
            [embeddings],
            ["embeddings"],
            [np.ndarray],
        )
        for document in documents:
            if type(document) not in self._supported_documents:
                raise TypeError("Provided document is not supported")

        try:
            self.vectorstore._collection.add(
                ids=[document.id for document in documents],
                embeddings=embeddings,
                documents=[f"{document.content}" for document in documents],
                metadatas=[document.metadata for document in documents],
            )
            logger.info("Successfully added new documents to vectorstore")
        except Exception as e:
            raise VectorStoreDocumentAdditionError(f"Error while adding new documents: {e}")

    @override
    def search(self, query: str, k: int = 5) -> list[BaseDocument]:
        """
        Performs a similarity search for a given query.

        Args:
            query: The query string to search for.
            k: The number of top results to return. Defaults to 5.

        Raises:
            TypeError: If query is not a string or k is not an integer.
            SimilaritySearchError: If similarity search fails.

        Returns:
            List of top-K similar documents, converted into supported `BaseDocument` types.
        """

        validate_dtypes([query, k], ["query", "k"], [str, int])

        try:
            self.retriever.search_kwargs["k"] = k
            relevant = self.retriever.invoke(query)
            documents = [
                self._to_base_documents[doc.metadata["type"]](doc) for doc in relevant
            ]
            return documents
        except Exception as e:
            raise VectorStoreSearchError(f"Error while searching for documents: {e}")

    @override
    def load(self, persist_dir: str) -> None:
        """
        Loads a Chroma vector store from the path and initializes it.

        Args:
            persist_dir: The path from which to load the vector store.

        Raises:
            TypeError: If persist_dir is not a string.
            VectoreStoreLoadingError: If loading of the vector store fails.
        """

        validate_dtypes([persist_dir], ["persist_path"], [str])

        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_func,
                persist_directory=persist_dir,
            )
            self.retriever = self.vectorstore.as_retriever(search_type=self.search_type)
            self.persist_dir = persist_dir
            logger.info("Successfully loaded vectorstore from disk")
        except Exception as e:
            raise VectorStoreLoadingError(
                f"Error while adding new supported document: {e}"
            )
