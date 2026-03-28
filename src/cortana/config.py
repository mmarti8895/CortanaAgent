from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
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
    stt_end_silence_seconds: float = Field(default=1.0, gt=0.0, le=5.0)
    stt_max_phrase_seconds: float = Field(default=8.0, gt=0.0, le=30.0)

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
    listen_timeout_seconds: float = Field(default=2.5, gt=0.0, le=10.0)
    listen_max_phrases: int = Field(default=6, ge=1, le=20)
    assistant_personality: str = (
        "You are Cortana, concise, warm, proactive, and highly tactical. "
        "Address the user with rotating military-style honorifics occasionally."
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: bool | str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        msg = f"Unsupported debug value: {value!r}"
        raise ValueError(msg)

    @field_validator("local_llm_model_path", "piper_model_path", "piper_config_path", mode="before")
    @classmethod
    def empty_path_to_none(cls, value: Path | str | None) -> Path | str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value


settings = CortanaSettings()
