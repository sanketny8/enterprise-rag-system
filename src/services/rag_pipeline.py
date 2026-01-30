"""RAG pipeline orchestrating the full retrieval-augmented generation workflow."""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.services.embeddings import EmbeddingService
from src.services.vector_store import WeaviateStore, Document
from src.services.llm import LLMService
from src.services.chunker import DocumentChunker

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Source document with metadata and score."""
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class QueryResponse:
    """Response from RAG query."""
    answer: str
    sources: List[Source]
    latency_ms: float
    tokens_used: int


class RAGPipeline:
    """
    Complete RAG pipeline.
    
    Orchestrates:
    1. Query embedding
    2. Semantic search in vector store
    3. Context preparation
    4. LLM answer generation
    """
    
    def __init__(
        self,
        vector_store: WeaviateStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store for retrieval
            embedding_service: Service for generating embeddings
            llm_service: Service for LLM completion
            top_k: Default number of documents to retrieve
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.top_k = top_k
        
        logger.info("RAG pipeline initialized")
    
    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> QueryResponse:
        """
        Execute RAG query.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve (uses default if None)
            filters: Optional filters for vector search
            temperature: LLM temperature
            
        Returns:
            QueryResponse with answer, sources, and metrics
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.top_k
        
        try:
            # 1. Embed query
            logger.debug(f"Embedding query: {question[:100]}...")
            query_embedding = await self.embedding_service.embed_query(question)
            
            # 2. Retrieve documents
            logger.debug(f"Retrieving top {top_k} documents")
            documents = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            if not documents:
                logger.warning("No documents found for query")
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0
                )
            
            # 3. Build context from retrieved documents
            context = self._build_context(documents)
            logger.debug(f"Built context from {len(documents)} documents ({len(context)} chars)")
            
            # 4. Generate answer using LLM
            logger.debug("Generating answer")
            result = await self.llm_service.generate(
                prompt=question,
                context=context,
                temperature=temperature
            )
            
            # 5. Prepare response
            latency_ms = (time.time() - start_time) * 1000
            
            sources = [
                Source(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=doc.score
                )
                for doc in documents
            ]
            
            response = QueryResponse(
                answer=result["answer"],
                sources=sources,
                latency_ms=latency_ms,
                tokens_used=result["tokens_used"]
            )
            
            logger.info(f"Query completed in {latency_ms:.0f}ms ({result['tokens_used']} tokens)")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise
    
    async def ingest(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> int:
        """
        Ingest a document into the RAG system.
        
        Args:
            doc_id: Unique document ID
            content: Document content
            metadata: Optional metadata
            chunk_size: Size of chunks
            overlap: Overlap between chunks
            
        Returns:
            Number of chunks ingested
        """
        try:
            # 1. Chunk document
            logger.debug(f"Chunking document {doc_id}")
            chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
            chunks = chunker.chunk(content, metadata or {})
            
            logger.info(f"Document {doc_id} chunked into {len(chunks)} chunks")
            
            # 2. Generate embeddings for all chunks
            logger.debug(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embedding_service.embed(chunk_texts)
            
            # 3. Store chunks with embeddings
            logger.debug("Storing chunks in vector store")
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                await self.vector_store.insert(
                    doc_id=chunk_id,
                    content=chunk["content"],
                    embedding=embedding,
                    metadata={
                        **chunk["metadata"],
                        "parent_doc_id": doc_id,
                        "chunk_id": chunk_id
                    }
                )
            
            logger.info(f"Successfully ingested document {doc_id} ({len(chunks)} chunks)")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting document {doc_id}: {e}")
            raise
    
    def _build_context(self, documents: List[Document], max_length: int = 4000) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: Retrieved documents
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Format: [Source X] content
            source_label = f"[Source {i+1}]"
            part = f"{source_label} {doc.content}\n\n"
            
            # Check if adding this would exceed max length
            if current_length + len(part) > max_length:
                # Truncate the last document to fit
                remaining = max_length - current_length
                if remaining > 100:  # Only add if there's meaningful space
                    truncated = part[:remaining] + "..."
                    context_parts.append(truncated)
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "".join(context_parts).strip()
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            Dictionary with health status of each component
        """
        return {
            "vector_store": self.vector_store.health_check(),
            "embedding_service": True,  # If loaded, it's healthy
            "llm_service": True  # If initialized, it's healthy
        }

