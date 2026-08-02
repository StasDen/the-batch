"""Defines custom exceptions for handling errors related to embedding operations."""

from exceptions.base import BaseException


class TextEmbeddingError(BaseException): ...


class ImageEmbeddingError(BaseException): ...
