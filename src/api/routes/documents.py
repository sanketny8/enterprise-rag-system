"""Document ingestion and management endpoints."""

import logging
from typing import Optional
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_rag_pipeline
from src.services.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


class Document(BaseModel):
    """Document model."""
    id: Optional[str] = None
    text: str
    metadata: dict = {}


class DocumentResponse(BaseModel):
    """Document response model."""
    id: str
    status: str
    chunks_created: int


@router.post("/documents", response_model=DocumentResponse)
async def ingest_document(
    document: Document,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> DocumentResponse:
    """
    Ingest a text document into the system.
    
    The document will be chunked, embedded, and stored in the vector database.
    """
    try:
        # Generate ID if not provided
        doc_id = document.id or str(uuid.uuid4())
        
        logger.info(f"Ingesting document: {doc_id}")
        
        # Ingest through pipeline
        chunks_created = await pipeline.ingest(
            doc_id=doc_id,
            content=document.text,
            metadata=document.metadata
        )
        
        logger.info(f"Successfully ingested document {doc_id} ({chunks_created} chunks)")
        
        return DocumentResponse(
            id=doc_id,
            status="ingested",
            chunks_created=chunks_created
        )
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> DocumentResponse:
    """
    Upload and ingest a file.
    
    Currently supports:
    - Plain text (.txt)
    - Future: PDF, DOCX, etc.
    """
    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        logger.info(f"Uploading file: {file.filename} as document {doc_id}")
        
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file.filename.endswith('.txt'):
            text = content.decode('utf-8')
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Currently only .txt files are supported."
            )
        
        # Ingest through pipeline
        chunks_created = await pipeline.ingest(
            doc_id=doc_id,
            content=text,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type
            }
        )
        
        logger.info(f"Successfully uploaded and ingested {file.filename} ({chunks_created} chunks)")
        
        return DocumentResponse(
            id=doc_id,
            status="ingested",
            chunks_created=chunks_created
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Document:
    """
    Retrieve a document by ID.
    
    Note: This returns the document metadata and chunks, not the original document.
    """
    try:
        # Search for chunks with this document ID
        # Use a dummy embedding since we're filtering by doc_id
        dummy_embedding = [0.0] * pipeline.embedding_service.get_dimension()
        
        documents = await pipeline.vector_store.search(
            query_embedding=dummy_embedding,
            top_k=100,  # Get all chunks
            filters={"parent_doc_id": document_id}
        )
        
        if not documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reconstruct document from chunks
        content = "\n\n".join([doc.content for doc in documents])
        metadata = documents[0].metadata if documents else {}
        
        return Document(
            id=document_id,
            text=content,
            metadata=metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> dict:
    """
    Delete a document and all its chunks from the system.
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Delete from vector store (this will delete all chunks)
        await pipeline.vector_store.delete(document_id)
        
        logger.info(f"Successfully deleted document {document_id}")
        
        return {"status": "deleted", "id": document_id}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

