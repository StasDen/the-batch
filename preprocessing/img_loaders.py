"""Defines interface and implementation for loading images from URLs."""

from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Optional
from io import BytesIO

import pydantic
from PIL import Image

from internals.utils import validate_dtypes, content_hash
from data.fetch import HttpxFetcher
from internals.logger import logger
from exceptions.fetch import FetchError
from exceptions.preprocessing import ImageLoadingError
from the_batch.batch_config import BATCH_ENDPOINT


class LoadedImage:
    """
    Wrapper for a loaded image with its source URL.

    Attributes:
        url: The URL from which the image was loaded.
        image: The loaded image object.
    """

    def __init__(self, url: str, image: Image.Image):
        self.url = url
        self.image = image

    def __str__(self):
        return f"LoadedImage (url={self.url})"


class ImageLoaderI(ABC):
    """Interface for image loaders."""

    @abstractmethod
    def load(
        img_url: str, handle_exceptions: bool, *args, **kwargs
    ) -> None | LoadedImage: ...


class HttpxImageLoader(pydantic.BaseModel, ImageLoaderI):
    """
    Image loader implementation that uses the `FetcherHttpx` to fetch image data
        and `PIL` to load the image from bytes.

    Attributes:
        fetcher: The fetcher instance used to get raw image bytes.

    Raises:
        FetchError: If error occurred while fetching data.
        ImageLoadingError: If error occurred while loading image data.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    fetcher: HttpxFetcher = pydantic.Field(default_factory=HttpxFetcher)
    seen_hashes: set[str] = pydantic.Field(default_factory=set)  # digests of images already loaded

    @override
    def load(
        self,
        img_url: str,
        handle_exceptions: bool = True,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        cookies: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[LoadedImage]:
        """
        Load an image from a given URL.

        Args:
            img_url: The HTTP/HTTPS URL of the image to fetch.
            handle_exception: Whether to suppress exceptions and return None on failure. Defaults to True.
            params: Query parameters to send with the HTTP request.
            headers: HTTP headers to send with the request.
            cookies: Cookies to send with the request.
            metadata: Additional metadata for the fetcher.

        Raises:
            ValueError: If URL does not start with '/' and `handle_exception` is False.
            FetchError: If fetching fails and `handle_exception` is False.
            ImageLoadingError: If image loading fails.

        Returns:
            LoadedImage or None: LoadedImage instance on success, None if failure and `handle_exception` is True.
        """

        validate_dtypes(
            [img_url],
            ["img_url"],
            [str],
        )
        if not img_url.startswith("/"):
            msg = f"Ivalid image src to fetch: {img_url}"
            if handle_exceptions:
                logger.error(msg)
                return None
            raise ValueError(msg)

        try:
            result = self.fetcher.fetch(
                url=f"{BATCH_ENDPOINT}{img_url}",
                params=params,
                headers=headers,
                cookies=cookies,
                metadata=metadata,
            )
            if not result.success:
                msg = "Error occurred while fetching data"
                if handle_exceptions:
                    logger.error(msg)
                    return None
                raise FetchError(msg)

            img_bytes = result.response.content
            digest = content_hash(img_bytes)
            if digest in self.seen_hashes:
                logger.info("Skipping already downloaded image...")
                return None
            self.seen_hashes.add(digest)

            image = LoadedImage(url=img_url, image=Image.open(BytesIO(img_bytes)))
            logger.info("Successfully loaded new image")
            return image
        except Exception as e:
            msg = f"Error occurred while loading data: {e}"
            if handle_exceptions:
                logger.error(msg)
                return None
            raise ImageLoadingError(msg)
