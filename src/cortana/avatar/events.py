from __future__ import annotations

from dataclasses import dataclass

from cortana.transport.udp import UdpJsonPublisher


@dataclass(slots=True)
class AvatarEmitter:
    publisher: UdpJsonPublisher

    async def speaking_state(self, speaking: bool) -> None:
        await self.publisher.send({"type": "speaking_state", "speaking": speaking})

    async def jaw_movement(self, amplitude: float) -> None:
        clamped = min(max(amplitude, 0.0), 1.0)
        await self.publisher.send({"type": "jaw_movement", "amplitude": clamped})

    async def gesture(self, name: str) -> None:
        await self.publisher.send({"type": "gesture", "name": name})
