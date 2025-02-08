from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    notion_api_key: str
    notion_database_id: str
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()