from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import os

from config.settings import EMBEDDING_MODEL

class EmbeddingModel:
    """Wrapper for text embedding models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """Initialize the embedding model."""
        self.model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        return self.model.embed_query(text)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        return self.model.embed_documents(documents)
    
    def get_model(self):
        """Get the underlying embedding model."""
        return self.model

def get_embedding_model():
    """Helper function to get embedding model instance."""
    model = EmbeddingModel()
    return model.get_model()