"""Configures and initializes BatchLLM with pre-defined prompt and vector store."""

from the_batch.llm.llm import BatchLLM
from the_batch.batch_config import prompt_template
from the_batch.vectorstore_pipeline import vectorstore

batch_llm = BatchLLM(vectorstore=vectorstore, prompt_template=prompt_template)
