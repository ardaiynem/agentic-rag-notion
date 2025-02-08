import chromadb
from chromadb.config import Settings
from config.settings import settings

class ChromaManager:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents, embeddings, ids):
        """
        Adds documents to the collection.

        Args:
            documents: List of document texts.
            embeddings: List of document embeddings.
            ids: List of document IDs.
        """
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    def delete(self, ids):
        """
        Deletes documents from the collection.

        Args:
            ids: List of document IDs.
        """
        self.collection.delete(ids)

    def query(self, query_embeddings, n_results):
        """
        Queries the collection for similar documents.

        Args:
            query_embeddings: Query embeddings.
            n_results: Number of results to return.
        Returns:
            List of document IDs and distances.
        """
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
    
    def count(self):
        """
        Returns the number of documents in the collection.
        
        Returns:
            Number of documents.
        """
        return self.collection.count()
    
    def peek(self, n=5):
        """
        Returns a preview of the collection.

        Args:
            n: Number of documents to return.
        Returns:
            List of document IDs and texts.
        """
        return self.collection.peek(n)