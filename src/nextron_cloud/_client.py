from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

from ._version import __version__
from .types import Account, Layout, ModelInfo

DEFAULT_BASE_URL = "https://nexcloud.physionlabs.ai"
RETRY_STATUS = (429, 502, 503, 504)
T = TypeVar("T")


class NexTronError(Exception):
    pass


class APIError(NexTronError):
    """A 4xx or 5xx answer. `code` is the server's short error name when it gives one."""

    def __init__(self, status_code: int, message: str, code: str | None = None, body: Any = None):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.code = code
        self.body = body


def _api_error(r: httpx.Response) -> APIError:
    try:
        body = r.json()
    except ValueError:
        return APIError(r.status_code, r.text or r.reason_phrase)
    if not isinstance(body, dict):
        return APIError(r.status_code, str(body), body=body)
    detail = body.get("detail")
    code = body.get("error")
    message = detail if isinstance(detail, str) else str(detail or code or r.reason_phrase)
    return APIError(r.status_code, message, code, body)


class NexTron:
    """Client for the NexCloud API. The key comes from `api_key` or NEXTRON_API_KEY."""

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, org: str | None = None,
                 timeout: float = 60.0, max_retries: int = 3):
        api_key = api_key or os.environ.get("NEXTRON_API_KEY")
        if not api_key:
            raise NexTronError("no API key: pass api_key or set NEXTRON_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "User-Agent": f"nextron-cloud-python/{__version__}"}
        if org:
            headers["X-Nextron-Org"] = org
        base_url = base_url or os.environ.get("NEXTRON_BASE_URL") or DEFAULT_BASE_URL
        url = httpx.URL(base_url)
        if url.scheme != "https" and url.host not in ("localhost", "127.0.0.1", "::1"):
            raise NexTronError(f"refusing to send the API key over {url.scheme}://; use https")
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._api_key = api_key
        self._timeout = timeout
        self.max_retries = max_retries

        from .checkpoints import Checkpoints
        from .datasets import Datasets
        from .jobs import Jobs
        from .inference import Inference
        from .rl import RL
        self.datasets = Datasets(self)
        self.checkpoints = Checkpoints(self)
        self.jobs = Jobs(self)
        self.rl = RL(self)
        self.inference = Inference(self)

    def me(self) -> Account:
        """The account the key belongs to, with its balance and the price list."""
        return Account.model_validate(self._get("/v1/me"))

    def models(self) -> list[ModelInfo]:
        """Every model a job can train, live: methods, longest sample, price, GPU setups and whether one is free now."""
        return [ModelInfo.model_validate(x) for x in self._get("/v1/models")]

    def layouts(self) -> list[Layout]:
        """The GPU setups, the models each trains, and whether one is free now."""
        return [Layout.model_validate(x) for x in self._get("/v1/layouts")]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> NexTron:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, *, retry: bool | None = None, **kwargs: Any) -> httpx.Response:
        """Send one request. GETs, and POSTs the caller marks safe to repeat, are retried on a dropped connection or a
        429/502/503/504."""
        if retry is None:
            retry = method == "GET"
        attempts = self.max_retries + 1 if retry else 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                r = self._http.request(method, path, **kwargs)
            except httpx.TransportError:
                if last:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code in RETRY_STATUS and not last:
                time.sleep(float(r.headers.get("Retry-After", 2**attempt)))
                continue
            if r.is_error:
                raise _api_error(r)
            return r
        raise AssertionError("unreachable")

    def _get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs).json()

    def _post(self, path: str, body: Any = None, **kwargs: Any) -> Any:
        return self._request("POST", path, json=body, **kwargs).json()


def poll(get: Callable[[], T], done: Callable[[T], bool], timeout: float | None, interval: float) -> T:
    """Call `get` every `interval` seconds until `done` holds; TimeoutError once `timeout` seconds have passed."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        value = get()
        if done(value):
            return value
        if deadline is None:
            time.sleep(interval)
            continue
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"still not done after {timeout} s")
        time.sleep(min(interval, left))
