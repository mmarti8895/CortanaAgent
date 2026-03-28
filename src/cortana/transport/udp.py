from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from typing import Any

import orjson


@dataclass(slots=True)
class UdpJsonPublisher:
    host: str
    port: int
    _sock: socket.socket = field(init=False)

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def send(self, message: dict[str, Any]) -> None:
        payload = orjson.dumps(message)
        loop = asyncio.get_running_loop()
        await loop.sock_sendto(self._sock, payload, (self.host, self.port))

    def close(self) -> None:
        self._sock.close()
