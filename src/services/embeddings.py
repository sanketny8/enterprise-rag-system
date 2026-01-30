"""Embedding service for generating text embeddings."""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses sentence-transformers for high-quality embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is a fast, lightweight model.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed([query])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

