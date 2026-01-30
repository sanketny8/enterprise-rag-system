"""Document chunking service."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunk documents into smaller pieces for better retrieval.
    
    Uses fixed-size chunking with overlap to preserve context.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            separator: Preferred separator for splitting (e.g., "\n\n" for paragraphs)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")
        
        logger.info(f"Initialized chunker: chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunks, each with content and metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Try to split by separator first (e.g., paragraphs)
        if self.separator in text:
            paragraphs = text.split(self.separator)
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                # If adding this paragraph would exceed chunk_size, save current chunk
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_index,
                            "chunk_size": len(current_chunk)
                        }
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + self.separator + para
                    chunk_index += 1
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += self.separator + para
                    else:
                        current_chunk = para
            
            # Add remaining chunk
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(current_chunk)
                    }
                })
        else:
            # Fallback: fixed-size chunking
            chunk_index = 0
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "start_pos": start,
                        "end_pos": end
                    }
                })
                
                # Move start position with overlap
                start += self.chunk_size - self.overlap
                chunk_index += 1
        
        logger.debug(f"Chunked text into {len(chunks)} chunks")
        return chunks
    
    def chunk_batch(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            texts: List of texts to chunk
            metadatas: Optional list of metadata dicts (one per text)
            
        Returns:
            List of all chunks from all documents
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")
        
        all_chunks = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # Add document index to metadata
            doc_metadata = {**metadata, "doc_index": i}
            chunks = self.chunk(text, doc_metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(texts)} documents into {len(all_chunks)} total chunks")
        return all_chunks

