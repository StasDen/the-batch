"""Defines configuration and result data classes for text parsing of HTML content."""

import pydantic

from data.parser_tag import TagI


class ParserConfig(pydantic.BaseModel):
    """
    Configuration container for text parser.

    Attributes:
        parsed_tags: A list of string identifiers corresponding to parsed tags extracted from text.

        tags: A list of tag configuration objects implementing the TagI interface.
              Each element provides detailed parsing instructions for a specific HTML tag or element.

    Raises:
        ValidationError:
            - If 'parsed_tags' or 'tags' is empty.
            - If both lists have different length.
            - If not all elements in 'tags' list has the same data type.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    tags: list[TagI]
    parsed_tags: list[str]

    @pydantic.model_validator(mode="after")
    @classmethod
    def validate_tags(cls, values):
        tags, parsed_tags = values.tags, values.parsed_tags
        tag_type = type(tags[0])
        if len(tags) == 0 or len(parsed_tags) == 0:
            raise ValueError("Tags cannot be empty")
        if len(tags) != len(parsed_tags):
            raise ValueError("Tag lists must have equal size")
        if not all(isinstance(tag, tag_type) for tag in tags):
            raise ValueError("All tags must have the same type")
        return values

    def __repr__(self):
        return f"ParserConfig (parsed_tags={self.parsed_tags}, tags={[tag_obj.tag for tag_obj in self.tags]})"

    def __iter__(self):
        return zip(self.tags, self.parsed_tags)


class ParsedData:
    """
    Structured result of parsed text from the given URL.

    Attributes:
        url: The source URL from which the text was parsed.
        parsed_tags: A list of string identifiers representing the tags.
        parsed_data:  A list of parsed string data corresponding to each tag.

    Dynamic Attributes:
        For each tag name in `parsed_tags`, an attribute is dynamically created on the instance
        with the tag name, holding the corresponding parsed data.
    """

    def __init__(self, url: str, parsed_tag_to_data: dict[str, list]):
        self.url = url
        self.parsed_tags = parsed_tag_to_data.keys()
        for parsed_tag, data in parsed_tag_to_data.items():
            setattr(self, parsed_tag.lower(), data)

    def __repr__(self):
        return f"ParsedData (url={self.url}, parsed_tags={self.parsed_tags})"

    def get_all(self) -> list:
        all_data = []
        for parsed_tag in self.parsed_tags:
            all_data.extend(getattr(self, parsed_tag.lower()))
        return all_data
