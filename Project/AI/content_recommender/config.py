"""Configuration loader. Reads API keys from .env and validates them."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime config. Loaded from .env (same directory as this file)."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_cse_id: str = Field(default="", alias="GOOGLE_CSE_ID")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    session_ttl_minutes: int = Field(default=30, alias="SESSION_TTL_MINUTES")

    def key_status(self) -> dict[str, bool]:
        """Which keys are populated (for /health endpoint)."""
        return {
            "gemini": bool(self.gemini_api_key),
            "google_api": bool(self.google_api_key),
            "google_cse": bool(self.google_cse_id),
        }


settings = Settings()
