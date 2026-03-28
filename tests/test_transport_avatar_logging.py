from __future__ import annotations

import asyncio
import socket

import pytest

from cortana.avatar.events import AvatarEmitter
from cortana.transport.udp import UdpJsonPublisher
from cortana.utils.logging import configure_logging, get_logger


@pytest.mark.asyncio
async def test_udp_send_and_avatar_events() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    host, port = receiver.getsockname()

    publisher = UdpJsonPublisher(host, port)
    avatar = AvatarEmitter(publisher)

    await avatar.speaking_state(True)
    data, _ = await asyncio.to_thread(receiver.recvfrom, 2048)
    assert b"speaking_state" in data

    await avatar.jaw_movement(1.5)
    data2, _ = await asyncio.to_thread(receiver.recvfrom, 2048)
    assert b'"amplitude":1.0' in data2

    await avatar.gesture("wave")
    data3, _ = await asyncio.to_thread(receiver.recvfrom, 2048)
    assert b"wave" in data3

    publisher.close()
    receiver.close()


def test_logging_configures() -> None:
    configure_logging(debug=True)
    logger = get_logger("unit")
    logger.info("ok")
