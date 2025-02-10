from fastapi import APIRouter, UploadFile
from app.main import app
from pydantic import BaseModel
from app.utils import loader, text_splitter
from app.embeddings.embedding import SentenceTransformerModel, OpenAIEmbeddingModel
from app.database import chroma_manager, notion_integration
from app.generation import generation
import os
import logging
from app.database import notion_integration
from app.rag.rag_pipeline import RAGPipeline

router = APIRouter()
vector_store = chroma_manager.ChromaManager()
# embedding_model = SentenceTransformerModel()
embedding_model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
response_generator = generation.ResponseGenerator()
# notion_manager = notion_integration.NotionManager()

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_document(file: UploadFile):
    """
    Uploads a document, processes it, and stores it in the vector store.

    Args:
        file: The document to upload.

    Returns:
        A JSON response indicating the status of the upload
    """

    # Save the uploaded file to a temporary location
    logger.info(f"Uploading file: {file.filename}")

    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())
    
    # PDF processing pipeline
    logger.info(f"Loading file: {file.filename}")
    pages = loader.load(temp_file_path)

    logger.info(f"Splitting text: {file.filename}")
    chunks = text_splitter.split_documents(pages)
    chunks_text = [chunk.page_content for chunk in chunks]
    
    # Embed and store
    logger.info(f"Embedding and storing {len(chunks)} chunks")
    embeddings = embedding_model.embed_documents(chunks_text)

    logger.info(f"Adding {len(chunks)} chunks to the vector store")
    vector_store.add(
        documents=chunks_text,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(chunks_text))]
    )
    
    # Remove the temporary file
    os.remove(temp_file_path)

    logger.info(f"Upload complete: {file.filename}")
    
    return {"status": "success", "chunks": len(chunks)}

@router.post("/query")
async def handle_query(request: QueryRequest):
    """
    Handles a user query by retrieving relevant documents from the vector store
    and generating a response using the RAG pipeline.

    Args:
        request: The user query.

    Returns:
        A JSON response containing the generated answer.
    """
    query = request.query

    rag_pipeline = RAGPipeline(embedding_model=embedding_model, vector_store=vector_store, response_generator=response_generator)

    # RAG pipeline
    result = rag_pipeline.process_query(query, None)
    return result

@router.get("/count")
async def get_count():
    """
    Returns the number of documents stored in the vector store.

    Returns:
        A JSON response containing the count of stored documents
    """
    return {"count": vector_store.count()}


app.include_router(router)