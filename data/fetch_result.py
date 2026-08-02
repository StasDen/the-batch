"""Defines data transfer objects (DTOs) for encapsulating results of website fetch operations."""

from typing import Optional, Protocol, Any, runtime_checkable
from datetime import datetime

import pydantic
import httpx


@runtime_checkable
class FetchResultI(Protocol):
    """
    Interface class for encapsulating the results of website fetch operations.

    This protocol defines the expected fields for a fetch result. All implementations
    should store metadata about the success or failure of a fetch operation.
    """

    def __init__(
        self,
        success: bool,
        url: str,
        response: Any,
        status_code: Optional[int],
        data: Optional[str],
        headers: Optional[dict],
        metadata: Optional[dict[str, Any]],
        timestamp: Optional[datetime],
        error: Optional[str],
    ) -> None:
        raise NotImplementedError


class FetchResult(pydantic.BaseModel):
    """
    DTO for the result of a website fetch operation.

    Attributes:
        success: Whether the fetch was successful.
        url: The URL that was fetched.
        responce: Request responce.
        data: The main body of the response in string format.
        status_code: HTTP status code returned by the server.
        headers: Response headers.
        error_message: Error details (if any).
        timestamp: Time when the fetch occurred.
        metadata: Additional metadata.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    success: bool
    url: str
    response: Optional[httpx.Response] = pydantic.Field(default=None, repr=False)
    status_code: Optional[int] = pydantic.Field(default=None, repr=False)
    data: Optional[str] = pydantic.Field(default=None, repr=False)
    metadata: Optional[dict[str, Any]] = pydantic.Field(default=None, repr=False)
    timestamp: datetime = pydantic.Field(default_factory=datetime.now, repr=False)
    error: Optional[str] = pydantic.Field(default=None, repr=False)

    def __str__(self):
        return f"""
        FetchResult(
            success={self.success}
            url={self.url}
            status_code={self.status_code}
            data={self.data}
            metadata={self.metadata}
            timestamp={self.timestamp}
            error={self.error}
        )
        """
