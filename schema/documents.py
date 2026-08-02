"""Defines structured document models for multimodal data with metadata support."""

from typing import Optional, runtime_checkable, ClassVar, Protocol, Any

import pydantic
from PIL import Image


@runtime_checkable
class BaseDocument(Protocol):
    """Base structured document."""

    id: str
    type: ClassVar
    content: str

    @property
    def metadata(self) -> dict[str, Any]: ...


class TextDocument(pydantic.BaseModel):
    """
    Structured text document.

    Attributes:
        id: A unique identifier for the document.
        type: The type of the document (fixed to 'text').
        content: Text content of the document.
        source_url: The source URL from which the text was extracted.
    """

    id: str
    type: ClassVar = "text"
    content: str
    source_url: Optional[str] = None

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "source_url": self.source_url,
        }


class ImageDocument(pydantic.BaseModel):
    """
    Structured image document.

    Attributes:
        id: A unique identifier for the image.
        type: The type of the document (fixed to 'image').
        content: Image description.
        image: The loaded image object.
        source_url: The source URL from which the image was extracted.
        image_url: The direct URL to the image content.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    id: str
    type: ClassVar = "image"
    content: str
    image: Optional[Image.Image] = pydantic.Field(default=None)
    image_url: str
    source_url: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "image_url": self.image_url,
            "source_url": self.source_url,
        }
