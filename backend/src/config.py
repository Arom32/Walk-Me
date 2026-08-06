from typing import Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_NAME: Optional[str] = None
    DB_PORT: str = "5432"
    DATABASE_URL: Optional[str] = None

    DRIVE_ASSETS_FOLDER_ID: str = "1EElriPe8ikT9rETt-oY1o5ZEijUty104"
    ASSETS_DIR: Optional[str] = None

    BASE_MODEL: str = "google/gemma-4-E2B-it"
    LORA_PATH: Optional[str] = None
    CHROMA_DIR: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), #".env"에서 해당 코드로 변경
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def get_db_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    def assets_dir(self) -> Path:
        if self.ASSETS_DIR:
            return Path(self.ASSETS_DIR).expanduser().resolve()
        return (BACKEND_DIR / "assets").resolve()

    def lora_path(self) -> Path:
        # 팀 공식 경로: ai/llm/lora_output/final (개인 Walk-Me-assets 학습분 사용 안 함)
        if self.LORA_PATH:
            return Path(self.LORA_PATH).expanduser().resolve()
        return (REPO_ROOT / "ai" / "llm" / "lora_output" / "final").resolve()

    def chroma_dir(self) -> Path:
        if self.CHROMA_DIR:
            return Path(self.CHROMA_DIR).expanduser().resolve()
        return (REPO_ROOT / "ai" / "rag" / "data" / "chroma").resolve()


settings = Settings()
