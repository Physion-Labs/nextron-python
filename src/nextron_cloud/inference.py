from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._client import NexTronError, poll
from .live import LiveRun
from .types import InferenceRun, RLEstimate

if TYPE_CHECKING:
    from ._client import NexTron


class Inference:
    """Inference runs: the model a training job saved, served on the GPUs of the cluster that trained it, so you can try
    it right after training and download it only when it is good. A job's model stays there 7 days. Call a run with
    `connect` (NexCloud's batch `sample`) or any OpenAI client (`openai_base_url`). Billed by time from ready to stopped."""

    def __init__(self, client: NexTron):
        self._client = client

    def estimate(self, *, job_id: str, max_hours: float) -> RLEstimate:
        """The price per GPU-hour, the run's GPUs, and what `max_hours` would hold; free."""
        return RLEstimate.model_validate(self._client._post(
            "/v1/inference/estimate", {"job_id": job_id, "max_hours": max_hours}, retry=True))

    def start(self, *, job_id: str, max_hours: float, idle_timeout: int | None = None, name: str | None = None,
              tags: builtins.list[str] | None = None) -> InferenceRun:
        """Serve the model of `job_id` (a training job that succeeded). The run stops itself after `max_hours`, or after
        `idle_timeout` seconds without a request (1800)."""
        settings = {"idle_timeout": idle_timeout, "name": name, "tags": tags}
        body = {"job_id": job_id, "max_hours": max_hours} | {k: v for k, v in settings.items() if v is not None}
        return InferenceRun.model_validate(self._client._post("/v1/inference", body))

    def list(self) -> list[InferenceRun]:
        return [InferenceRun.model_validate(x) for x in self._client._get("/v1/inference")]

    def get(self, run_id: str) -> InferenceRun:
        return InferenceRun.model_validate(self._client._get(f"/v1/inference/{run_id}"))

    def wait(self, run_id: str, *, timeout: float | None = None, poll_interval: float = 15,
             on_poll: Callable[[InferenceRun], None] | None = None) -> InferenceRun:
        """Wait until the model is loaded (about 8 minutes). NexTronError if the run ended instead."""
        def get() -> InferenceRun:
            run = self.get(run_id)
            if on_poll:
                on_poll(run)
            return run
        run = poll(get, lambda r: r.state not in ("queued", "starting"), timeout, poll_interval)
        if run.state != "ready":
            raise NexTronError(f"{run_id} {run.state}" + (f": {run.error}" if run.error else ""))
        return run

    def stop(self, run_id: str, *, wait: bool = False, timeout: float | None = None, poll_interval: float = 10) -> InferenceRun:
        """Stop the run: billing stops at once and the GPUs are released about a minute later."""
        run = InferenceRun.model_validate(self._client._post(f"/v1/inference/{run_id}/stop", retry=True))
        if wait:
            run = poll(lambda: self.get(run_id), lambda r: r.done, timeout, poll_interval)
        return run

    def connect(self, run_id: str) -> LiveRun:
        """`sample` on a ready run: a whole batch of prompts in one call, as in an RL run (it has no `train`)."""
        return LiveRun(self._client, run_id, base=f"/v1/inference/{run_id}/nextron/v1")

    def openai_base_url(self, run_id: str) -> str:
        """The base URL for an OpenAI client (`chat.completions`, `completions`, `models`); its api_key is your NexCloud
        key. Any model name is the run's model. Through the public API one call must finish in about 90 seconds; use
        `connect` for large batches."""
        return f"{str(self._client._http.base_url).rstrip('/')}/v1/inference/{run_id}/v1"
