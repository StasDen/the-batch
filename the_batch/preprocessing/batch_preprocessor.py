"""Module for preprocessing text and images from The Batch website using extraction, splitting and description modules."""

from typing import Any

import pydantic

from preprocessing.text_exctraction import TextExtractorI, BS4TextExtractor
from preprocessing.img_loaders import ImageLoaderI, HttpxImageLoader
from preprocessing.text_splitter import TextSplitterI, RecursiveTextSplitter
from preprocessing.img_describer import ImageDescriberI, BLIPImageDescriber
from schema.documents import TextDocument, ImageDocument


class TheBatchPreprocessor(pydantic.BaseModel):
    """
    Preprocessor for extracting, splitting, loading and describing multimodal data from The Batch.

    This class integrates text and image preprocessing steps to prepare multimodal data for downstream use
    (e.g., RAG, embedding models, etc.). It relies on custom components on each step.

    Attributes:
        text_extractor:
            Component responsible for extracting text from parsed HTML elements.
            Defaults to `BS4TextExtractor`.

        text_splitter:
            Component responsible for splitting extracted text into manageable chunks for downstream tasks.
            Defaults to `RecursiveTextSplitter`.

        image_loader:
            Component responsible for downloading and loading images from URLs.
            Defaults to `HttpxImageLoader`.

        image_describer:
            Component responsible for generating natural language descriptions of images.
            Defaults to `BLIPImageDescriber`.

    Raises:
        ValidationError: If attributes does not match extecped data type.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    text_extractor: TextExtractorI = pydantic.Field(default=BS4TextExtractor())
    image_loader: ImageLoaderI = pydantic.Field(default=HttpxImageLoader())
    text_splitter: TextSplitterI = pydantic.Field(default=RecursiveTextSplitter())
    image_describer: ImageDescriberI = pydantic.Field(default=BLIPImageDescriber())

    def preprocess(
        self, source_url: str, elements: list[Any], img_urls: list[str]
    ) -> list[TextDocument | ImageDocument]:
        """
        Executes the preprocessing steps for multimodal data.

        Args:
            source_url: The source URL of the fetched content.
            elements: Parsed HTML elements containing text to be extracted.
            img_urls: List of image URLs to download and described.

        Returns:
            A list containing `TextDocuments` and `ImageDocuments`.
        """

        extracted = self.text_extractor.extract(elements=elements)
        splitted = self.text_splitter.split(text=extracted, source_url=source_url)

        images = [self.image_loader.load(url) for url in img_urls]
        images = [image for image in images if image]
        descriptions = [self.image_describer.describe(image) for image in images]

        final_documents = splitted + descriptions
        return final_documents
