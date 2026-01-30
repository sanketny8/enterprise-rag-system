"""Vector store implementation using Weaviate."""

import logging
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.auth import AuthApiKey
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class WeaviateStore:
    """
    Vector store using Weaviate.
    
    Handles document storage, embedding storage, and semantic search.
    """
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        class_name: str = "Documents"
    ):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            api_key: Optional API key for authentication
            class_name: Name of the Weaviate class/collection
        """
        self.url = url
        self.class_name = class_name
        
        logger.info(f"Connecting to Weaviate at {url}")
        
        if api_key:
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=AuthApiKey(api_key)
            )
        else:
            self.client = weaviate.Client(url=url)
        
        # Create schema if it doesn't exist
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure the schema exists in Weaviate."""
        schema = {
            "class": self.class_name,
            "description": "Documents for RAG retrieval",
            "vectorizer": "none",  # We provide our own vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Source of the document"
                },
                {
                    "name": "doc_id",
                    "dataType": ["string"],
                    "description": "Document ID"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Index of chunk within document"
                }
            ]
        }
        
        try:
            # Check if class exists
            existing_schema = self.client.schema.get()
            class_names = [c["class"] for c in existing_schema.get("classes", [])]
            
            if self.class_name not in class_names:
                logger.info(f"Creating schema for class: {self.class_name}")
                self.client.schema.create_class(schema)
            else:
                logger.info(f"Schema for class {self.class_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")
    
    async def insert(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Insert a document with its embedding.
        
        Args:
            doc_id: Unique document ID
            content: Document content
            embedding: Document embedding vector
            metadata: Additional metadata
            
        Returns:
            UUID of the inserted object
        """
        metadata = metadata or {}
        
        data_object = {
            "content": content,
            "doc_id": doc_id,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", 0)
        }
        
        try:
            uuid = self.client.data_object.create(
                data_object=data_object,
                class_name=self.class_name,
                vector=embedding
            )
            logger.debug(f"Inserted document {doc_id} with UUID {uuid}")
            return uuid
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Semantic search using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filters (e.g., {"source": "doc_1"})
            
        Returns:
            List of matching documents with scores
        """
        try:
            query = (
                self.client.query
                .get(self.class_name, ["content", "doc_id", "source", "chunk_index"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(top_k)
                .with_additional(["distance", "id"])
            )
            
            # Add filters if provided
            if filters:
                where_filter = {
                    "operator": "And",
                    "operands": [
                        {
                            "path": [key],
                            "operator": "Equal",
                            "valueString": value
                        }
                        for key, value in filters.items()
                    ]
                }
                query = query.with_where(where_filter)
            
            result = query.do()
            
            # Parse results
            documents = []
            if "data" in result and "Get" in result["data"]:
                for item in result["data"]["Get"].get(self.class_name, []):
                    # Distance to similarity score (lower distance = higher similarity)
                    distance = item.get("_additional", {}).get("distance", 0)
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    doc = Document(
                        id=item.get("_additional", {}).get("id", ""),
                        content=item.get("content", ""),
                        metadata={
                            "doc_id": item.get("doc_id", ""),
                            "source": item.get("source", ""),
                            "chunk_index": item.get("chunk_index", 0)
                        },
                        score=similarity
                    )
                    documents.append(doc)
            
            logger.debug(f"Found {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def delete(self, doc_id: str):
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
        """
        try:
            # Find all objects with this doc_id
            result = (
                self.client.query
                .get(self.class_name, ["doc_id"])
                .with_where({
                    "path": ["doc_id"],
                    "operator": "Equal",
                    "valueString": doc_id
                })
                .with_additional(["id"])
                .do()
            )
            
            # Delete each object
            if "data" in result and "Get" in result["data"]:
                for item in result["data"]["Get"].get(self.class_name, []):
                    uuid = item.get("_additional", {}).get("id")
                    if uuid:
                        self.client.data_object.delete(uuid, self.class_name)
                        logger.debug(f"Deleted object {uuid}")
            
            logger.info(f"Deleted document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Weaviate is healthy."""
        try:
            self.client.schema.get()
            return True
        except Exception as e:
            logger.error(f"Weaviate health check failed: {e}")
            return False

