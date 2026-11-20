import json
from collections.abc import Callable

import httpx
import pytest

from nextron_cloud import NexTron

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], NexTron]:
    """A client whose API calls and storage calls (presigned URLs) all go to `handler`."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def make(handler: Handler) -> NexTron:
        transport = httpx.MockTransport(handler)
        real = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=transport, **kw))
        return NexTron("nk_test", base_url="https://api.test")
    return make


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def mp4(width: int, height: int, seconds: float, fps: int = 10) -> bytes:
    """An H.264 MP4 file of blank frames."""
    import io

    import av
    out = io.BytesIO()
    with av.open(out, mode="w", format="mp4") as target:
        stream = target.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for n in range(int(seconds * fps)):
            frame = av.VideoFrame(width, height, "yuv420p")
            frame.pts = n
            target.mux(stream.encode(frame))
        target.mux(stream.encode())
    return out.getvalue()
