"""Dependency injection for API routes."""

import logging
from functools import lru_cache
from typing import Optional

from src.services.embeddings import EmbeddingService
from src.services.vector_store import WeaviateStore
from src.services.llm import LLMService
from src.services.rag_pipeline import RAGPipeline
from src.config import settings

logger = logging.getLogger(__name__)


# Global instances (initialized once)
_rag_pipeline: Optional[RAGPipeline] = None


@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the RAG pipeline instance.
    
    This is cached and reused across requests for efficiency.
    """
    global _rag_pipeline
    
    if _rag_pipeline is None:
        logger.info("Initializing RAG pipeline")
        
        # Initialize services using settings from config
        embedding_service = EmbeddingService(
            model_name=settings.default_embedding_model
        )
        
        vector_store = WeaviateStore(
            url=settings.weaviate_url,
            api_key=settings.weaviate_api_key if settings.weaviate_api_key else None,
            class_name=settings.weaviate_index_name
        )
        
        llm_service = LLMService(
            provider=settings.default_llm_provider,
            api_key=settings.openai_api_key if settings.openai_api_key else None,
            model=settings.default_llm_model
        )
        
        # Create pipeline
        _rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_service=llm_service,
            top_k=settings.top_k
        )
        
        logger.info("RAG pipeline initialized successfully")
    
    return _rag_pipeline


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    pipeline = get_rag_pipeline()
    return pipeline.embedding_service


async def get_vector_store() -> WeaviateStore:
    """Get vector store instance."""
    pipeline = get_rag_pipeline()
    return pipeline.vector_store


async def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    pipeline = get_rag_pipeline()
    return pipeline.llm_service

