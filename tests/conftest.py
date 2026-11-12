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
