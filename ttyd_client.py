import asyncio
import websockets
import config


class TtydClient:
    def __init__(self, url=None):
        self._url = url or config.TTYD_URL
        self._ws = None
        self._loop = None

    async def _connect(self):
        self._ws = await websockets.connect(self._url)
        try:
            await asyncio.wait_for(self._ws.recv(), timeout=1.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

    async def _send_text(self, text):
        if not self._ws or self._ws.closed:
            await self._connect()
        for char in text:
            await self._ws.send(b"\x00" + char.encode("utf-8"))
            await asyncio.sleep(0.005)

    async def _send_enter(self):
        if not self._ws or self._ws.closed:
            await self._connect()
        await self._ws.send(b"\x00\r")

    def connect(self):
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._connect())

    def send_text(self, text):
        if not self._loop:
            self.connect()
        self._loop.run_until_complete(self._send_text(text))

    def send_enter(self):
        if not self._loop:
            self.connect()
        self._loop.run_until_complete(self._send_enter())

    def close(self):
        if self._ws and self._loop:
            self._loop.run_until_complete(self._ws.close())
        if self._loop:
            self._loop.close()
            self._loop = None
