from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CortanaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Cortana"
    debug: bool = False

    udp_host: str = "127.0.0.1"
    udp_port: int = 8765

    wake_word: str = "cortana"
    wake_threshold: int = Field(default=85, ge=50, le=100)

    stt_model_size: str = "small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_sample_rate: int = 16000
    stt_chunk_seconds: float = 1.25

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_timeout_seconds: float = 25.0

    local_llm_enabled: bool = True
    local_llm_model_path: Path | None = None

    tts_backend: str = "piper"
    piper_executable: str = "piper"
    piper_model_path: Path | None = None
    piper_config_path: Path | None = None

    memory_max_turns: int = 8
    assistant_personality: str = (
        "You are Cortana, concise, warm, proactive, and highly tactical. "
        "Address the user with rotating military-style honorifics occasionally."
    )


settings = CortanaSettings()
