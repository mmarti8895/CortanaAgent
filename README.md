# Cortana Agent

A production-grade, local-first real-time voice assistant framework implemented in Python 3.11+.

## Features

- Async modular architecture with microservice-style boundaries.
- Wake word detection (`Cortana`) with configurable sensitivity.
- Streaming STT (faster-whisper + VAD) with terminal fallback.
- Streaming LLM responses using OpenAI API with resilient local fallback.
- Real-time TTS (Piper backend) with console fallback.
- UDP JSON avatar events (`speaking_state`, `jaw_movement`, `gesture`).
- Extensible command plugin system.
- Structured logging for debug/prod.
- Cross-platform scripts for Ubuntu/Windows setup.

## Architecture Overview

```text
Audio In -> STT Stream -> WakeWord Gate -> Orchestrator -> Plugin Router -> LLM
                                                            |             |
                                                            v             v
                                                         Command      TTS Stream -> Audio Out
                                                                            |
                                                                            v
                                                                       UDP Avatar Events
```

Core modules:

- `cortana.core.orchestrator`: state machine and routing.
- `cortana.audio.stt`: streaming STT backends.
- `cortana.llm.engine`: OpenAI stream + local fallback.
- `cortana.audio.tts`: Piper or console speech output.
- `cortana.plugins`: intent/plugin framework.
- `cortana.avatar.events` + `cortana.transport.udp`: avatar UDP emitter.

## Installation

### Linux (Ubuntu)

```bash
./scripts/setup_linux.sh
```

### Windows 11 (PowerShell)

```powershell
./scripts/setup_windows.ps1
```

## Running

1. Copy and edit environment:

```bash
cp .env.example .env
```

2. Start assistant:

```bash
python -m cortana
```

When microphone dependencies are unavailable, the app automatically switches to text input mode for development.

## Configuration Modes

- `config/local.env` for local-only mode.
- `config/api.env` for OpenAI + Piper mode.

Example:

```bash
set -a; source config/local.env; set +a
python -m cortana
```

## UDP Avatar Protocol

The assistant sends JSON datagrams to `UDP_HOST:UDP_PORT`:

- `{"type": "speaking_state", "speaking": true|false}`
- `{"type": "jaw_movement", "amplitude": 0.0-1.0}`
- `{"type": "gesture", "name": "..."}`

## Testing

```bash
pytest
```

Coverage threshold is set to 90% for the `cortana` package.

## Extending Commands

Add a plugin implementing `CommandPlugin` and register it in `build_orchestrator()`.

## Performance Notes

- Keep STT model in memory and chunk around `1.0-1.5s` for low latency.
- Use `int8` compute type on CPU for strong speed/quality tradeoff.
- Tune `WAKE_THRESHOLD` to reduce false positives.
