# LLM
## llm.py
This module defines BatchLLM, a multimodal RAG system that
integrates an Ollama LLM, a prompt template and a vector store to answer user queries with both text and relevant image documents. 
It manages initialization, querying and error handling.

## llm_configured.py
This module sets up and stores a ready-to-use BatchLLM instance, 
configured with the prompt template and vector store for efficient querying and retrieval in the app.


# Preprocessing
## batch_data_loader.py
This module integrates a customizable data ingestion system for The Batch website, 
combining fetchers and parsers (e.g., HttpxFetcher, BS4Parser) with configurable parsing rules 
to load and structure multimodal data (text, images) for downstream processing.

## batch_preprocessor.py
This module combines multimodal data preprocessing steps, including text extraction and splitting from HTML elements, image loading and describing into a unified framework for preparing data from The Batch website. It leverages custom components for each step (e.g., BS4 text extractor, BLIP image describer) to produce structured TextDocument and ImageDocument outputs for downstream application.


# Store
## the_batch_image_documents_store.json
JSON file that serves as the persistence location for the 
image documents extracted and preprocessed from The Batch site.

## the_batch_vectorestore_persist_dir
Directory that serves as the persistence location for the Choroma vector store used by The Batch system. This vector store contains both text and image documents extracted and preprocessed from The Batch site.

## the_batch_urls.txt
Paths for storage of article URLs from The Batch.


# batch_config.py
This module defines the configuration setup for The Batch app, including:

- Paths for storage of URLs, vectorstores and image documents.

- Fetcher and parser instances for web data retrieval and HTML parsing.

- A detailed ParserConfig specifying tag mappings for text and image extraction from The Batch website.

- A specialized PromptTemplate for the model to generate explicit, comprehensive answers based strictly on provided context, without external knowledge or references.

# batch_exceptions.py
This module defines custom exception specific to The Batch LLM system.

# vectorestore_pipeline.py
This module handles the creation and loading of a Chroma vectorstore for The Batch dataset, integrating:

- Data ingestion: Loads TheBatch URLs and fetches their content.

- Preprocessing: Processes HTML content and extracts structured data (both text and image).

- Image document management: Saves mappings of image documents to a JSON store.

- Embedding: Generates documents embeddings using a SentenceTransformerTextEmbedding model.

- Chroma vectorstore: Stores preprocessed documents and embeddings in a persistent vectorstore for semantic search and RAG use cases.

# batch_app.py
Streamlit-based multimodal news assistant that allows users to ask questions and get responses with relevant text and images from The Batch site using a pretrained LLM and an image document store. It maintains a chat interface, displays messages, handles image retrieval and offers error handling with chat reset functionality.
