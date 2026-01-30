"""Query and RAG inference endpoints."""

import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.dependencies import get_rag_pipeline
from src.services.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: int = 5
    use_reranking: bool = True
    stream: bool = False
    filters: Optional[dict] = None
    temperature: float = 0.7


class Source(BaseModel):
    """Source document model."""
    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Source]
    latency_ms: float
    tokens_used: int


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """
    Execute a RAG query.
    
    Retrieves relevant documents and generates an answer using LLM.
    """
    try:
        logger.info(f"Received query: {request.query[:100]}...")
        
        response = await pipeline.query(
            question=request.query,
            top_k=request.top_k,
            filters=request.filters,
            temperature=request.temperature
        )
        
        return QueryResponse(
            answer=response.answer,
            sources=[
                Source(
                    content=source.content,
                    metadata=source.metadata,
                    score=source.score
                )
                for source in response.sources
            ],
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_rag_stream(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> StreamingResponse:
    """
    Execute a RAG query with streaming response.
    
    Streams the generated answer token-by-token.
    """
    async def generate():
        try:
            # First, retrieve context
            query_embedding = await pipeline.embedding_service.embed_query(request.query)
            documents = await pipeline.vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k,
                filters=request.filters
            )
            
            if not documents:
                yield f"data: {json.dumps({'error': 'No relevant documents found'})}\n\n"
                return
            
            # Build context
            context = pipeline._build_context(documents)
            
            # Stream LLM response
            async for chunk in pipeline.llm_service.generate_stream(
                prompt=request.query,
                context=context,
                temperature=request.temperature
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send sources at the end
            sources_data = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in documents
            ]
            yield f"data: {json.dumps({'sources': sources_data})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

