"""Defines an image describer interface and its implementation."""

from typing import Optional
from abc import ABC, abstractmethod
from typing_extensions import override

import pydantic
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from internals.logger import logger
from internals.utils import validate_dtypes, generate_unique_doc_id
from schema.documents import ImageDocument
from exceptions.preprocessing import ImageDescriptionError
from preprocessing.img_loaders import LoadedImage


class ImageDescriberI(ABC):
    """Interface class for image describer."""

    @abstractmethod
    def describe(img: LoadedImage) -> ImageDocument: ...


class BLIPImageDescriber(pydantic.BaseModel, ImageDescriberI):
    """
    Generate image descriptions with Hugging Face BLIP model.

    Attributes:
        pretrained_model_name_or_path: Name or path of the pretrained BLIP model to load.
        processor: Optional BLIP processor. If None, it is loaded from HF.
        model: Optional BLIP model. If None, it is loaded from HF.

    Raises:
        ImageDescriptionError: If image description fails.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    processor: Optional[BlipProcessor] = pydantic.Field(default=None, repr=False)
    model: Optional[BlipForConditionalGeneration] = pydantic.Field(
        default=None, repr=False
    )
    pretrained_model_name_or_path: str = "Salesforce/blip-image-captioning-base"

    def model_post_init(self, context) -> None:
        if not self.processor:
            try:
                self.processor = BlipProcessor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    use_fast=True,
                    device_map="cuda:0",
                )
            except Exception as e:
                raise ImageDescriptionError(f"BlipProcessor initialization failed: {e}")
        if not self.model:
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.pretrained_model_name_or_path,
                    device_map="cuda:0",
                    max_memory={0: 8192},
                )
            except Exception as e:
                raise ImageDescriptionError(f"BlipModel initialization failed: {e}")

    @override
    def describe(self, image: LoadedImage) -> ImageDocument:
        """
        Generates a description for the given image.

        Args:
            image: An image object containing the image data (`PIL` Image) and the source URL
                    Instance of `LoadedImage` class.

        Raises:
            TypeError: If image is not of `LoadedImage` type.

        Returns:
            ImageDocument: Structured image document with image description.
        """

        validate_dtypes(
            inputs=[image],
            input_names=["image"],
            required_dtypes=[LoadedImage],
        )

        try:
            inputs = self.processor(images=image.image, return_tensors="pt").to(
                "cuda:0"
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            description = self.processor.decode(outputs[0], skip_special_tokens=True)

            logger.info("Description is successfully generated")
            return ImageDocument(
                id=generate_unique_doc_id(content=description),
                content=description,
                image=image.image,
                source_url=image.url,
                image_url=image.url,
            )
        except Exception as e:
            raise ImageDescriptionError(
                f"Error happened while generating description: {e}"
            )
