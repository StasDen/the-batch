"""Streamlit-based multimodal news assistant querying TheBatch LLM and displaying relevant text and images."""

import streamlit as st

from the_batch.llm.llm_configured import batch_llm
from internals.logger import logger
from internals.utils import load_img_docs_from_json
from the_batch.batch_config import THE_BATCH_IMAGE_DOCUMENTS_STORE

img_documents = load_img_docs_from_json(THE_BATCH_IMAGE_DOCUMENTS_STORE)


def main():
    st.set_page_config("The Batch Multimodal News Assistant", layout="wide")
    st.title("The Batch Multimodal News Assistant")
    st.markdown(
        """
                Welcome to the **The Batch Multimodal News Assistant**!

                Ask a question and receive an answer from relevent articles and images from TheBatch site.
        """
    )

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["images"]:
                images = [img_documents[img.id] for img in msg["images"]]
                st.caption(
                    "Be careful: images may not be entirely relevant to your query."
                )
                for retrieved_img in images:
                    st.image(retrieved_img.image)

    if user_query := st.chat_input("Your question"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
                "images": None,
            }
        )
        try:
            with st.spinner("Generating response..."):
                response = batch_llm.query(user_query)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.text_response,
                        "images": response.img_response,
                    }
                )
        except Exception as e:
            logger.error(f"Error occurred while generating response: {e}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Something went wrong on my end - please try again.",
                    "images": None,
                }
            )
        st.rerun()

    with st.sidebar:
        if st.button("Delete history"):
            st.session_state.messages.clear()
            st.rerun()


if __name__ == "__main__":
    main()
