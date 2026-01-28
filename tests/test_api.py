"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "enterprise-rag-system"
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_query_endpoint():
    """Test query endpoint."""
    request_data = {
        "query": "What is RAG?",
        "top_k": 5,
        "use_reranking": True
    }
    response = client.post("/api/v1/query", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data

