"""Provides an interface and a httpx-based implementation for fetching website content."""

from typing import Optional
from abc import ABC, abstractmethod
from typing_extensions import override

import httpx

from data.fetch_result import FetchResult
from internals.logger import logger
from internals.utils import validate_dtypes

Timeout = int | float | tuple[int | float, int | float]


class FetcherI(ABC):
    """Interface for website fetcher."""

    @abstractmethod
    def fetch(self, url, *args, **kwargs) -> FetchResult: ...


class HttpxFetcher(FetcherI):
    """
    Fetch data from a website using `httpx` library.

    Args:
        timeout: Maximum amount of time (in seconds) the request will wait for.

    Raises:
        TypeError: If timeout is not a single int (float) number or tuple with two int (float) numbers.
    """

    def __init__(self, timeout: Timeout = 10):
        self._validate_timeout(timeout)
        self.timeout = timeout

    def __repr__(self):
        return f"FetcherHttpx (timeout={self.timeout})"

    @staticmethod
    def _validate_timeout(timeout: Timeout):
        if isinstance(timeout, (int, float)):
            return
        if (
            isinstance(timeout, tuple)
            and len(timeout) == 2
            and all(isinstance(i, (int, float)) for i in timeout)
        ):
            return
        raise TypeError(
            "'timeout' must be int (float) or tuple[int, float] (tuple[float, int])"
        )

    @override
    def fetch(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        cookies: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> FetchResult:
        """
        Fetch data from given URL.

        Args:
            url: Request URL.
            params: Dictionary to send in the request query.
            headers: Custom HTTP headers to send.
            cookies: Cookies to send with the request.
            metadata: Additional metadata.

        Raises:
            TypeError: If any of the arguments does not match expected data type.

        Returns:
            FetchResult: The result of a fetch.
        """
        
        validate_dtypes(
            [url, params, headers, cookies, metadata],
            ["url", "params", "headers", "cookies", "metadata"],
            [
                str,
                dict | type(None),
                dict | type(None),
                dict | type(None),
                dict | type(None),
            ],
        )

        try:
            response = httpx.get(
                url,
                params=params,
                headers=headers,
                cookies=cookies,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            return FetchResult(
                success=True,
                url=url,
                status_code=response.status_code,
                response=response,
                data=response.text,
                headers=response.headers,
                metadata=metadata,
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error while fetching data from {url}: {e}")
            return FetchResult(
                success=False, url=url, error="Request timed out.", metadata=metadata
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while fetching data from {url}: {e}")
            return FetchResult(
                success=False,
                url=url,
                status_code=response.status_code,
                response=response,
                error=e,
                headers=e.response.headers,
                metadata=metadata,
            )
        except httpx.RequestError as e:
            logger.error(f"Request error while fetching data from {url}: {e}")
            return FetchResult(success=False, url=url, error=e, metadata=metadata)
        except Exception as e:
            logger.error(f"Unexpected error while fetching data from {url}: {e}")
            return FetchResult(success=False, url=url, error=e, metadata=metadata)
