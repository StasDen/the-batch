"""Defines interface and implementation for parsing structured data from fetched website content."""

from typing import Literal
from typing_extensions import override
from abc import ABC, abstractmethod

import pydantic
import bs4

from data.fetch_result import FetchResultI, FetchResult
from data.parser_config import ParserConfig, ParsedData
from data.parser_tag import BS4Tag
from internals.logger import logger
from exceptions.parse import BS4ParsingError
from exceptions.fetch import FetchError
from internals.utils import validate_dtypes


class ParserI(ABC):
    """Interface for text parsers that parse structured data from raw website content."""

    @abstractmethod
    def parse(self, fetch_result: FetchResultI, config: ParserConfig) -> str: ...


class Bs4Parser(pydantic.BaseModel, ParserI):
    """
    Implementation of ParserI with BeautifulSoup4.

    Attributes:
        parser: The backend parser to use with BeautifulSoup. Defaults to 'html.parser'.

    Raises:
        ValidationError: If specified backend parser is invalid.
    """

    parser: Literal["html.parser", "lxml", "html5lib"] = pydantic.Field(
        default="html.parser"
    )

    @override
    def parse(self, fetch_result: FetchResult, config: ParserConfig) -> ParsedData:
        """
        Parses raw content from HTML using BeautifulSoup4 based on the provided configuration.

        Args:
            fetch_result: FetchResultI object containing fetching results.
            parser_config: Configuration defining mappings of field names to BS4Tag objects.

        Raises:
            FetchError: If data fetch was unsuccessful.
            TypeError: If any tag in parser_config.tags is not of type BS4Tag.
            BS4ParsingError: If data parsing fails.
        """

        validate_dtypes(
            [fetch_result, config],
            ["fetch_result", "config"],
            [FetchResult, ParserConfig],
        )
        if not fetch_result.success:
            raise FetchError("Cannot parse data from failed request")
        if not all(isinstance(tag, BS4Tag) for tag in config.tags):
            raise TypeError("All tags must be of 'BS4Tag' type")
        
        try:
            soup = bs4.BeautifulSoup(markup=fetch_result.data, features=self.parser)
            parsed_tag_to_data = {}
            for tag, parsed_tag in config:
                name, attrs, recursive, limit = tag.construct().values()
                parsed_tag_to_data[parsed_tag] = soup.find_all(
                    name=name, attrs=attrs, recursive=recursive, limit=limit
                )
            logger.info("Successfully parsed response data")
            return ParsedData(
                url=fetch_result.url, parsed_tag_to_data=parsed_tag_to_data
            )
        except Exception as e:
            raise BS4ParsingError(f"Error while parsing the data: {e}")
