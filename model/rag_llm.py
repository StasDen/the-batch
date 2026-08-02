""" "Defines the RAG LLM interface and implementation."""

from typing import runtime_checkable, Protocol

import pydantic
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.llms.ollama import Ollama  # don't use OllamaLLM from langchain_ollama because of errors
import numpy as np

from schema.documents import BaseDocument
from store.vectorstore import VectorStoreI
from model.llm_response import RAGLLMResponse
from exceptions.model import RagLLMInitializationError


@runtime_checkable
class RagLLMI(Protocol):
    """Interface class for RAG LLM model."""

    def __init__(self, vectorstore: VectorStoreI, *args, **kwargs): ...

    def clear_vectorstore(self) -> None: ...

    def update_vectorstore(
        self, documents: list[BaseDocument], embeddings: np.ndarray
    ) -> None: ...

    def search_vectorstore(self, k: int = 5) -> list[BaseDocument]: ...

    def query(self, user_query: str, k: int = 5) -> str: ...


class OllamaRAGLLM(pydantic.BaseModel):
    """
    OllamaRAGLLM is a RAG model wrapper that includes:
        - A vector store for document retrieval.
        - A prompt template to inject context into queries.
        - An Ollama LLM to generate responses.

    The model retrieves relevant documents from a vector store based on user query,
    constructs a context, and invokes the LLM with new formatted prompt.

    Attributes:
        model: Language model. Defaults to Llama 3.2.
        prompt_template: Template with input variables ('context', 'user_query').
        vectorstore: Vectorstore that implements VectorStoreI interface.

    Raises:
        RagLLMInitializationError: If OllamaRAGLLM initialization fails.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    vectorstore: VectorStoreI
    model: Ollama = pydantic.Field(default=Ollama(model="llama3.2"))
    prompt_template: PromptTemplate

    def _get_context(self, documents: list[BaseDocument]) -> str:
        """
        Constructs the context string by concatenating content from the retrieved documents.

        Args:
            documents: Retrieved documents.

        Returns:
            Concatenated text content of relevant documents.
        """

        return "".join(document.content for document in documents)

    @pydantic.model_validator(mode="after")
    def validate_prompt(cls, values) -> None:
        """
        Validates that the prompt template contains the required input variables.

        Raises:
            ValueError: If the prompt template input is not {'context', 'user_query'}.
        """

        prompt_template = values.prompt_template
        if set(prompt_template.input_variables) != {"context", "user_query"}:
            raise RagLLMInitializationError(
                "Prompt must contain 'context' and 'query'."
            )
        return values

    def clear_vectorstore(self) -> None:
        """Clears the vector store, removing all stored documents and embeddings."""

        self.vectorstore.clear()

    def update_vectorstore(
        self, documents: list[BaseDocument], embeddings: np.ndarray
    ) -> None:
        """
        Updates the vector store with new documents and their embeddings.

        Args:
            documents: Documents to add.
            embeddings: Corresponding vector embeddings.
        """

        self.vectorstore.add_documents(docs=documents, embeddings=embeddings)

    def search_relevant(self, query: str, k: int = 5) -> list[BaseDocument]:
        """
        Retrieves the top-k relevant documents from the vector store for a given user query.

        Args:
            user_query: The user's query.
            k: Number of top documents to retrieve. Defaults to 5.

        Returns:
            The top-k relevant documents.
        """

        return self.vectorstore.search(query, k)

    def query(self, user_query: str, k: int = 5) -> RAGLLMResponse:
        """
        Executes a retrieval-augmented query:
        1. Retrieves top-k documents relevant to query.
        2. Constructs the context from the documents.
        3. Formats the prompt with context and user query.
        4. Invokes the Ollama model to generate a response.

        Args:
            user_query: The user's query.
            k: Number of relevant documents to retrieve. Defaults to 5.

        Returns:
            RAGLLMResponse: Response object containing the user query, model output and relevant documents.
        """

        relevant_docs = self.search_relevant(user_query, k)
        context = self._get_context(relevant_docs)
        user_prompt = self.prompt_template.invoke(
            {"user_query": user_query, "context": context}
        )
        response = self.model.invoke(user_prompt)

        llm_response = RAGLLMResponse(
            user_query=user_query, llm_response=response, relevant_docs=relevant_docs
        )
        return llm_response
