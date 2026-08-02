# Multimodal RAG System for The Batch website

A Multimodal RAG (Retrieval-Augmented Generation) system that retrieves and summarizes relevant news articles from The Batch, leveraging both textual and visual data. Supports query-based search, question answering and relevant media display.

## Description
This project implements a Multimodal RAG system designed to:

- Retrieve relevant articles from The Batch using advanced retrieval techniques.

- Incorporate textual and visual data (e.g., images from the articles).

- Answer user queries using a generative model enhanced by retrieved content.

- Display associated media alongside article summaries.

The system integrates:

- Dense retrievers (textual) and visual retrievers (image embeddings).

- Generative models for summarization and question-answering.

- User interface for interactive querying and media display.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

**IMPORTANT:** you need Ollama app and pulled Llama 3.2 model to proceed.

## Usage
Make sure you have Nvidia GPU with at least 8 GB of VRAM. Change device mappings otherwise.

To start Ollama server for the Multimodal RAG System, run the following commands*:

```bash
ollama serve
ollama run llama3.2
```

*The second command must be used in different terminal tab.

To start the interactive user interface, run the Streamlit app with the command below:

```bash
streamlit run batch_app.py
```

This will lanch the web-based interface where you can enter you queries and view article summaries along with relevant images.

You might see a message about LangChain library deprecation in the logs. Suggested library caused errors so have not been used during development (though it can be fixed in the future).

Tip: set `CREATE_VECTORESTORE` to `False` once vector store is created.

## TODO
- Implement automated tests for both backend and UI components.
- Add support for PNG, JPEG, SVG image formats in multimodal retrieval.
- Implement a modular framework for creating and managing sophisticated UI components.
