"""Defines app system, including fetchers, parsers, tag mappings and vectorstore paths."""

from pathlib import Path

from langchain_core.prompts.prompt import PromptTemplate

from data.fetch import HttpxFetcher
from data.parser import Bs4Parser
from data.parser_config import ParserConfig
from data.parser_tag import BS4Tag

BASE_DIR = Path(__file__).resolve().parent

THE_BATCH_URLS_PATH = (BASE_DIR / "store" / "the_batch_urls.txt").as_posix()
THE_BATCH_VECTORESTORE_PERSIST_DIR = (
    BASE_DIR / "store" / "the_batch_vectorestore_persist_dir"
).as_posix()
THE_BATCH_IMAGE_DOCUMENTS_STORE = (
    BASE_DIR / "store" / "the_batch_image_documents_store.json"
).as_posix()

BATCH_ENDPOINT = "https://www.deeplearning.ai"
COLLECTION_NAME = "the_batch"

CREATE_VECTORESTORE = False

parser = Bs4Parser()
fetcher = HttpxFetcher()

the_batch_parser_config = ParserConfig(
    parsed_tags=[
        "article_title",
        "publication_date",
        "author_name",
        "main_content",
        "images",
        "captions",
        "tags",
        "parahraph",
    ],
    tags=[
        BS4Tag(tag="h1"),
        BS4Tag(tag="time"),
        BS4Tag(tag="span", attrs={"class": "author"}),
        BS4Tag(tag="div", recursive=True),
        BS4Tag(tag="img"),
        BS4Tag(tag="figcaption"),
        BS4Tag(tag="a", attrs={"class": "tag"}),
        BS4Tag(tag="p", recursive=True),
    ],
)

prompt_template = PromptTemplate(
    input_variables=["context", "user_query"],
    template="""
You are an advanced AI assistant. 
Your task is to provide a **detailed and explicit** response to the question **based strictly on the information provided in the context below**.
You MUST NOT incorporate any information beyond what is given in the context. You MUST NOT reference or mention the context explicitly or state that the information is from a source.
Your answer must be **as detailed, comprehensive, and explicit as possible** based solely on the provided context, covering all relevant details.

Context:
{context}

Question:
{user_query}

Instructions:
- Provide a detailed, explicit, and comprehensive answer strictly using the information from the context.
- DO NOT reference or mention the context or its presence.
- If the context does not contain enough information to answer the question, respond with:
"The information provided from TheBatch is insufficient to fully answer this question."
""",
)
