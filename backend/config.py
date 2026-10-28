import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Playlist & Directories
    PLAYLIST_URL: str = "https://www.youtube.com/playlist?list=PLDzeHZWIZsTryvtXdMr6rPh4IDexB5NIA"
    DATA_DIR: str = "data"

    # Chunking
    CHUNK_SIZE_SECONDS: int = 60
    CHUNK_OVERLAP_SECONDS: int = 0

    # Speech to Text
    STT_PROVIDER: str = "faster-whisper"
    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_CPU_THREADS: int = 4
    GROQ_API_KEY: str = ""

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # Vector DB
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "love_babbar_dsa"
    QDRANT_PATH: str = "data/qdrant_db"

    # LLM
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-3.8-flash"
    LLM_API_KEY: str = ""

    # Retrieval
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.28
    EXPAND_ADJACENT_CHUNKS: bool = True

    # Server
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000

    @property
    def data_path(self) -> Path:
        p = BASE_DIR / self.DATA_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def manifests_dir(self) -> Path:
        p = self.data_path / "manifests"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def audio_dir(self) -> Path:
        p = self.data_path / "audio"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def transcripts_dir(self) -> Path:
        p = self.data_path / "transcripts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chunks_dir(self) -> Path:
        p = self.data_path / "chunks"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def qdrant_path(self) -> Path:
        p = BASE_DIR / self.QDRANT_PATH
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def videos_json_path(self) -> Path:
        return self.manifests_dir / "videos.json"

    @property
    def pipeline_manifest_path(self) -> Path:
        return self.manifests_dir / "pipeline_manifest.json"

settings = Settings()
