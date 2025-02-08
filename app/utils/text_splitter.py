from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document]):
    """
    Splits a list of documents into chunks using a recursive character-based text splitter.

    Args:
        documents: List of documents to split.

    Returns:
        List of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

