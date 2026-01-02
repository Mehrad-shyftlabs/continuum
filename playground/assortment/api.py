"""
FastAPI endpoint for Assortment API.

Basic FastAPI setup for assortment management.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Assortment API",
    description="API for assortment management",
    version="1.0.0",
)

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Assortment API is running")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Assortment API is healthy")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8089)
