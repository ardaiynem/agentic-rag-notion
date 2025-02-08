from abc import ABC, abstractmethod
import time
from config.settings import settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.utils.text_splitter import split_texts

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

class OpenAIEmbeddingModel:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.api_tpm_limit = 1000000  # Max tokens per minute (TPM) limit
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            # Split texts into smaller chunks
            batch_max_tokens = 100000  # Max tokens per request
            text_chunks = split_texts(texts, max_tokens=batch_max_tokens)
            embeddings = []
            for chunk in text_chunks:
                response = self.client.embeddings.create(
                    input=chunk,
                    model=self.model_name,
                )
                embeddings.extend([item.embedding for item in response.data])

                # Avoid exceeding the API limit
                sleep_time = 60 / (self.api_tpm_limit // batch_max_tokens)
                time.sleep(sleep_time + 5)  # Add a small buffer for safety

            return embeddings
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

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