"""Defines custom exceptions for errors encountered during data preprocessing operations."""

from exceptions.base import BaseException


class ImageLoadingError(BaseException): ...


class ImageDescriptionError(BaseException): ...


class TextExctractingError(BaseException): ...


class TextSplittingError(BaseException): ...


class TextSplitterInitializationError(BaseException): ...
