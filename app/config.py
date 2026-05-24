"""
Application configuration — loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str = Field(..., description="Groq API key")
    LLM_MODEL: str = Field("llama3-8b-8192", description="Groq model to use")
    LLM_TEMPERATURE: float = Field(0.0, ge=0.0, le=1.0)

    # Chunking
    CHUNK_SIZE: int = Field(512, description="Chars per chunk")
    CHUNK_OVERLAP: int = Field(64, description="Overlap between consecutive chunks")

    # Retrieval
    RETRIEVER_TOP_K: int = Field(4, description="Number of chunks to retrieve per query")
    MEMORY_WINDOW_K: int = Field(5, description="Number of past turns to keep in memory")

    # Session management
    SESSION_TTL_SECONDS: int = Field(3600, description="Session expiry time in seconds")
    MAX_SESSIONS: int = Field(100, description="Max concurrent sessions before eviction")

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Incremental indexing
    ENABLE_INCREMENTAL_INDEX: bool = Field(True, description="Only re-embed changed files")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
