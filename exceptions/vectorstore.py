"""Defines custom exceptions for errors encountered during vector store operations."""

from exceptions.base import BaseException


class VectorStoreDocumentAdditionError(BaseException): ...


class VectorStoreSearchError(BaseException): ...


class VectorStoreInitializationError(BaseException): ...


class VectorStoreLoadingError(BaseException): ...


class VectorStoreClearingError(BaseException): ...
