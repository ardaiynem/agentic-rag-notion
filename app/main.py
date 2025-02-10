from fastapi import FastAPI
from config.settings import settings
from app.database.chroma_manager import ChromaManager
from app.embeddings.embedding import SentenceTransformerModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = SentenceTransformerModel()
chroma = ChromaManager()

@app.on_event("startup")
async def initialize():
    # Warm up embedding model
    health_check_embedding_result = embeddings.embed_query("Hello, world!")
    print(f"Embedding model initialized: {health_check_embedding_result}")
    # Verify ChromaDB connection
    health_check_client_result = chroma.client.heartbeat()
    print(f"ChromaDB client initialized: {health_check_client_result}")