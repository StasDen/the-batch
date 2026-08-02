"""Provides an interface and implementation for splitting large text into smaller chunks."""

from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import override

import pydantic
from langchain_text_splitters import RecursiveCharacterTextSplitter

from internals.utils import validate_dtypes, generate_unique_doc_id
from exceptions.preprocessing import TextSplittingError, TextSplitterInitializationError
from schema.documents import TextDocument


class TextSplitterI(ABC):
    """Interface class for text splitter."""

    @abstractmethod
    def split(self, text: str, source_url: str) -> list[TextDocument]: ...


class RecursiveTextSplitter(pydantic.BaseModel, TextSplitterI):
    """
    A text splitter implementation using LangChain `RecursiveCharacterTextSplitter`
        that splits input text into chunks and returns a list of TextDocument instances.

    Attributes:
        chunk_size: The maximum number of characters per chunk.
        chunk_overlap: Number of characters from the end of a chunk that will be repeated at the start of the next one.
        separators: List of string delimiters to try when breaking the text.
        splitter: Instance of `RecursiveCharacterTextSplitter` class.

    Raises:
        ValidationError: If attributes does not match excpected data types.
        TextSplitterInitializationError: If RecursiveTextSplitter initialization fails.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    separators: Optional[list[str]] = pydantic.Field(default=None)
    chunk_size: int = pydantic.Field(default=1500)
    chunk_overlap: int = pydantic.Field(default=0)
    splitter: Optional[RecursiveCharacterTextSplitter] = pydantic.Field(
        default=None, repr=False
    )

    def model_post_init(self, context):
        if not self.separators:
            self.separators = ["\n\n", "\n", " ", ""]
        if not self.splitter:
            try:
                self.splitter = RecursiveCharacterTextSplitter(
                    separators=self.separators,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            except Exception as e:
                raise TextSplitterInitializationError(
                    f"Error when initializing splitter: {e}"
                )

    @override
    def split(self, text: str, source_url: str) -> list[TextDocument]:
        """
        Split the input text into chunks and wrap each chunk in a TextDocument.

        Args:
            text: The input text to split.
            source_url: URL of the text source.

        Raises:
            TypeError: If text or source_url is not a string.
            TextSplittingError: If text splitting fails.

        Returns:
            List of text chunks wrapped as TextDocuments.
        """

        validate_dtypes([text, source_url], ["text", "source_url"], [str, str])

        try:
            splitted = self.splitter.split_text(text)
            documents = [
                TextDocument(
                    id=generate_unique_doc_id(
                        content=s, metadata={"source_url": source_url}
                    ),
                    content=s,
                    source_url=source_url,
                )
                for s in splitted
            ]
            return documents
        except Exception as e:
            raise TextSplittingError(f"Error occurred while splitting data: {e}")
