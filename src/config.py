"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = "enterprise-rag-system"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # Local LLM
    vllm_url: str = "http://localhost:8001"
    vllm_model: str = "meta-llama/Llama-2-7b-chat-hf"

    # Model Settings
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4-turbo-preview"
    default_embedding_model: str = "text-embedding-3-small"
    default_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Vector Database
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""
    weaviate_index_name: str = "Documents"

    # Qdrant (alternative)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "documents"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl: int = 3600

    # ElasticSearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_api_key: str = ""
    elasticsearch_index: str = "documents"

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_system"

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunk_size: int = 1024
    top_k: int = 5
    rerank_top_n: int = 3
    similarity_threshold: float = 0.7
    use_hybrid_search: bool = True
    hybrid_search_alpha: float = 0.5

    # LLM Generation
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream_response: bool = True

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    token_budget_per_user: int = 100000

    # Caching
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95

    # Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    enable_tracing: bool = False

    # Prometheus
    prometheus_enabled: bool = True
    metrics_port: int = 9090

    # Security
    secret_key: str = "change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Document Processing
    max_file_size_mb: int = 50
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".html"]
    )
    extract_images: bool = False
    ocr_enabled: bool = False

    # Evaluation
    eval_dataset_path: str = "data/eval/test_qa.json"
    eval_metrics: List[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()

