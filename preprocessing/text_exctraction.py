"""Provides an interface and implementation for extracting and processing text from HTML tags."""

from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import override

import pydantic
import bs4

from internals.utils import validate_dtypes
from exceptions.preprocessing import TextExctractingError


class TextExtractorI(ABC):
    """Interface class for text extractor."""

    @abstractmethod
    def extract(self, elements: list[Any]) -> str: ...


class BS4TextExtractor(pydantic.BaseModel, TextExtractorI):
    """
    Extract text from BeautifulSoup Tags.

    Attributes:
        separator: Used to join parts of text within a tag.
        strip: Whether to strip whitespace from each text part.
        join_symbol: Used to join extracted text.
    """

    strip: bool = pydantic.Field(default=True)
    separator: str = pydantic.Field(default=" ")
    join_symbol: str = "\n"

    @override
    def extract(self, elements: list[bs4.element.Tag]) -> str:
        """
        Extract text from a list of `bs4.element.Tag` objects.

        Args:
            elements: List of BeautifulSoup tags.

        Raises:
            TypeError: If `elements` is not a list or contains non-tag objects.
            TextExtractionError: If text extraction fails.

        Returns:
            str: Extracted and concatenated text.
        """

        validate_dtypes([elements], ["elements"], [list])

        text = []
        try:
            for tag in elements:
                validate_dtypes([tag], ["tag"], [bs4.element.Tag])
                text.append(tag.get_text(separator=self.separator, strip=self.strip))
            result = self.join_symbol.join(text)
            return result
        except Exception as e:
            raise TextExctractingError(f"Error occurred while extracting data: {e}")
