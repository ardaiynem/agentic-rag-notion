from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

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

def count_tokens(texts, model_name="text-embedding-3-small"):
    """
    Counts the number of tokens in a list of texts for a given model.

    Args:
        texts: List of texts.
        model_name: Model name.

    Returns:
        Total number of tokens.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return sum(len(encoding.encode(text)) for text in texts)

def split_texts(texts, max_tokens=500000):
    """
    Splits a list of texts into chunks of approximately equal token counts.

    Args:
        texts: List of texts.
        max_tokens: Maximum number of tokens per chunk.

    Returns:
        List of chunks.
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    for text in texts:
        token_count = count_tokens([text])
        if current_token_count + token_count > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0
        current_chunk.append(text)
        current_token_count += token_count

    if current_chunk:
        chunks.append(current_chunk)
    return chunks
