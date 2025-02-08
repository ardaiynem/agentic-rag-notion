from abc import ABC, abstractmethod
# from langchain.embeddings import OpenAIEmbeddings
from config.settings import settings
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents into vectors.

        Args:
            texts: List of documents to embed.
        Returns:
            List of embeddings (each a list of floats).
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a query into a vector.

        Args:
            text: Query to embed.
        Returns:
            Embedding vector as a list of floats.
        """
        pass

# class OpenAIEmbeddingModel(EmbeddingModel):
#     def __init__(self):
#         self.model = OpenAIEmbeddings(
#             model="text-embedding-3-small",  # Latest and cost-effective
#             openai_api_key=settings.openai_api_key
#         )
    
#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         return self.model.embed_documents(texts)
    
#     def embed_query(self, text: str) -> list[float]:
#         return self.model.embed_query(text)


class SentenceTransformerModel(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        # self.model.max_seq_length = 512  # Optimize for chunk size

    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts)
    
    def batch_embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, 
            batch_size=64,
            convert_to_numpy=False,
            normalize_embeddings=True
        ).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0]