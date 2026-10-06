"""Manages The Batch Chroma vectorstore with preprocessed text and image documents."""

import numpy as np

from the_batch.batch_config import (
    THE_BATCH_URLS_PATH,
    THE_BATCH_IMAGE_DOCUMENTS_STORE,
    COLLECTION_NAME,
    THE_BATCH_VECTORESTORE_PERSIST_DIR,
    CREATE_VECTORESTORE,
)
from schema.documents import ImageDocument
from store.chroma import ChromaVectorStore
from embedding.text_embedding import SentenceTransformerTextEmbedding
from internals.utils import save_img_docs
from internals.adapters import ChromaVectorStoreAdapter
from the_batch.preprocessing.batch_data_loader import BatchDataLoader
from the_batch.preprocessing.batch_preprocessor import TheBatchPreprocessor


def create_batch_vectorstore():
    # load urls
    with open(THE_BATCH_URLS_PATH, "r") as f:
        batch_urls = f.readlines()
    batch_urls = [url.strip() for url in batch_urls]

    # load data
    batch_data_loader = BatchDataLoader()
    load_data = [batch_data_loader.load(url) for url in batch_urls]

    # preprocess data
    preprocessor = TheBatchPreprocessor()
    documents = []
    for data in load_data:
        img_urls = [img["src"] for img in data.images]

        prep_document = preprocessor.preprocess(
            img_urls=img_urls, elements=data.get_all(), source_url=data.url
        )
        documents.extend(prep_document)

    # save image documents
    img_documents = {doc.id: doc for doc in documents if isinstance(doc, ImageDocument)}
    save_img_docs(img_documents, THE_BATCH_IMAGE_DOCUMENTS_STORE)

    # embed documents
    embeddings_adapter = ChromaVectorStoreAdapter(
        embedding_func=SentenceTransformerTextEmbedding()
    )
    embeddings = np.vstack(
        [embeddings_adapter.embed_query(doc.content) for doc in documents]  # TODO: use dwt coeffs
    )

    # create vectorstore and add documents
    vectorstore = ChromaVectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=THE_BATCH_VECTORESTORE_PERSIST_DIR,
        embedding_func=embeddings_adapter,
    )
    vectorstore.add_documents(documents, embeddings)

    return vectorstore


def load_vectorstore():
    embeddings_adapter = ChromaVectorStoreAdapter(
        embedding_func=SentenceTransformerTextEmbedding()
    )
    vectorstore = ChromaVectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=THE_BATCH_VECTORESTORE_PERSIST_DIR,
        embedding_func=embeddings_adapter,
    )
    vectorstore.load(persist_dir=THE_BATCH_VECTORESTORE_PERSIST_DIR)

    return vectorstore


if not CREATE_VECTORESTORE:
    vectorstore = load_vectorstore()
else:
    vectorstore = create_batch_vectorstore()
