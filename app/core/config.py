
from pathlib import Path
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Research & Knowledge Assistant"
    environment: str = "development"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # LLM
    groq_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dimensions: int = Field(default=384, gt=0)

    # Chunking
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # Retrieval
    retrieval_top_k: int = Field(default=4, gt=0)

    # Storage
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    chroma_collection: str = "research_documents"

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )
        return self


settings = Settings()
