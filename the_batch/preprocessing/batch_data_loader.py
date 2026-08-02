"""Defines fetching and parsing data from The Batch website using custom fetchers and parsers."""

import pydantic

from data.fetch import FetcherI, HttpxFetcher
from data.parser import ParserI, Bs4Parser, ParsedData
from data.parser_config import ParserConfig
from the_batch.batch_config import the_batch_parser_config


class BatchDataLoader(pydantic.BaseModel):
    """
    Loader for retrieving and parsing data from The Batch website.

        This class integrates fetching and parsing logic using custom modules,
        enabling seamless loading of multimodal data (text, images) from The Batch site.

    Attributes:
        fetcher:
            An instance responsible for fetching raw data from The Batch site.
            Defaults to `HttpxFetcher`.

        parser:
            An instance responsible for parsing fetched HTML content.
            Defaults to `BS4Parser`.

        parsing_config:
            Configuration object defining tag mappings and parsing rules specific to The Batch.
            Defaults to predefined `the_batch_parser_config`.

    Raises:
        ValidationError: If attributes does not match expected data type.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    fetcher: FetcherI = pydantic.Field(default=HttpxFetcher())
    parser: ParserI = pydantic.Field(default=Bs4Parser())
    parsing_config: ParserConfig = pydantic.Field(default=the_batch_parser_config)

    def load(self, doc_url: str) -> ParsedData:
        """Fetches HTML content from the specified URL and parses it into a structured format."""

        fetch_result = self.fetcher.fetch(doc_url)
        parsed = self.parser.parse(fetch_result, self.parsing_config)
        return parsed
