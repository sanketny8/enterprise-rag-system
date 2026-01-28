"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness check for Kubernetes."""
    # Check vector DB, Redis, etc.
    return {"ready": True}

