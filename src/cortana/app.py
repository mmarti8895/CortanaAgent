from __future__ import annotations

import asyncio

from cortana.audio.stt import BufferedWhisperSTT, TextInputSTT
from cortana.audio.tts import ConsoleTTS, PiperTTS, TextToSpeechError
from cortana.audio.wakeword import WakeWordDetector
from cortana.config import settings
from cortana.core.models import ConversationMemory
from cortana.core.orchestrator import Orchestrator
from cortana.llm.engine import HonorificRotator, LocalFallbackEngine, OpenAiLlmEngine
from cortana.plugins.builtin import GreetingPlugin, TimeCommandPlugin
from cortana.plugins.manager import PluginManager
from cortana.transport.udp import UdpJsonPublisher
from cortana.avatar.events import AvatarEmitter
from cortana.utils.logging import configure_logging, get_logger


def build_orchestrator() -> Orchestrator:
    configure_logging(settings.debug)
    logger = get_logger("cortana.bootstrap")

    publisher = UdpJsonPublisher(settings.udp_host, settings.udp_port)
    avatar = AvatarEmitter(publisher=publisher)

    try:
        stt = BufferedWhisperSTT(
            model_size=settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            sample_rate=settings.stt_sample_rate,
            chunk_seconds=settings.stt_chunk_seconds,
        )
    except RuntimeError:
        logger.warning("stt.fallback", backend="text-input")
        stt = TextInputSTT()

    honorifics = HonorificRotator()
    fallback = LocalFallbackEngine(honorifics=honorifics)
    openai_api_key = None
    if settings.openai_api_key is not None:
        candidate_api_key = settings.openai_api_key.get_secret_value().strip()
        if candidate_api_key:
            openai_api_key = candidate_api_key

    if openai_api_key is None:
        llm = fallback
    else:
        llm = OpenAiLlmEngine(
            api_key=openai_api_key,
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            personality=settings.assistant_personality,
            fallback=fallback,
        )

    if (
        settings.tts_backend == "piper"
        and settings.piper_model_path is not None
    ):
        # Validate required TTS dependencies (e.g., sounddevice) before constructing PiperTTS.
        try:
            import sounddevice  # type: ignore[unused-import]
        except ImportError:
            logger.warning("tts.fallback", backend="console", reason="sounddevice missing")
            tts = ConsoleTTS(avatar=avatar)
        else:
            try:
                tts = PiperTTS(
                    executable=settings.piper_executable,
                    model_path=str(settings.piper_model_path),
                    config_path=str(settings.piper_config_path) if settings.piper_config_path else None,
                    avatar=avatar,
                )
            except TextToSpeechError:
                logger.warning("tts.fallback", backend="console")
                tts = ConsoleTTS(avatar=avatar)
    else:
        tts = ConsoleTTS(avatar=avatar)

    plugins = PluginManager()
    plugins.register(TimeCommandPlugin())
    plugins.register(GreetingPlugin())

    return Orchestrator(
        stt=stt,
        tts=tts,
        llm=llm,
        wakeword=WakeWordDetector(settings.wake_word, settings.wake_threshold),
        plugins=plugins,
        memory=ConversationMemory(max_turns=settings.memory_max_turns),
    )


async def main() -> None:
    orchestrator = build_orchestrator()
    await orchestrator.run()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
