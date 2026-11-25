from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._client import NexTronError, poll
from .live import LiveRun
from .types import RLEstimate, RLRollouts, RLRun, RLStep

if TYPE_CHECKING:
    from ._client import NexTron


class RL:
    """Serverless RL: an RL run is a model on a node of GPUs that your own loop samples from and trains, step by
    step. NexCloud runs every sample and every optimizer step. Drive one with `connect` (NexCloud's own calls) or with
    the `tinker` SDK and tinker-cookbook (`env`). Billed by time from ready to stopped."""

    def __init__(self, client: NexTron):
        self._client = client

    def estimate(self, *, model: str, max_hours: float) -> RLEstimate:
        """The price per GPU-hour, the run's GPUs, and what `max_hours` would hold; free."""
        return RLEstimate.model_validate(self._client._post(
            "/v1/training/sessions/estimate", {"model": model, "max_hours": max_hours}, retry=True))

    def start(self, *, model: str, max_hours: float, max_context: int | None = None, max_grad_norm: float | None = None,
              grad_clip: bool | None = None, idle_timeout: int | None = None, images: bool = False,
              video_fps: float | None = None, video_max_frames: int | None = None, video_frame_tokens: int | None = None,
              from_job: str | None = None, name: str | None = None, tags: builtins.list[str] | None = None) -> RLRun:
        """Queue an RL run. It stops itself after `max_hours`, or after `idle_timeout` seconds without a sample or a
        training step. `max_context` (tokens) bounds a prompt with its reply, and a training sequence. Gradients are
        clipped at `max_grad_norm` (1.0); `grad_clip=False` turns clipping off, which makes a step about 2 s shorter.
        With `images` its prompts may hold images (the models whose `input` lists "image"), and videos sampled as
        `video_fps`, `video_max_frames` and `video_frame_tokens` say (the model's defaults when unset; see the guide).
        With `from_job` it starts from the model that training job saved instead of the base model."""
        settings = {"from_job": from_job, "max_context": max_context, "max_grad_norm": max_grad_norm, "grad_clip": grad_clip,
                    "idle_timeout": idle_timeout, "images": images or None, "video_fps": video_fps,
                    "video_max_frames": video_max_frames, "video_frame_tokens": video_frame_tokens, "name": name, "tags": tags}
        body: dict[str, Any] = {"model": model, "max_hours": max_hours}
        body.update({key: value for key, value in settings.items() if value is not None})
        return RLRun.model_validate(self._client._post("/v1/training/sessions", body))

    def list(self) -> list[RLRun]:
        return [RLRun.model_validate(x) for x in self._client._get("/v1/training/sessions")]

    def get(self, run_id: str) -> RLRun:
        return RLRun.model_validate(self._client._get(f"/v1/training/sessions/{run_id}"))

    def wait(self, run_id: str, *, timeout: float | None = None, poll_interval: float = 15,
             on_poll: Callable[[RLRun], None] | None = None) -> RLRun:
        """Wait until the run is ready (its model is loaded: 6-8 minutes, about 20 for qwen3.8-next-180b). NexTronError if it ended instead.
        `on_poll` sees every poll's run, e.g. to print progress."""
        def get() -> RLRun:
            run = self.get(run_id)
            if on_poll:
                on_poll(run)
            return run
        run = poll(get, lambda r: r.state not in ("queued", "starting"), timeout, poll_interval)
        if run.state != "ready":
            raise NexTronError(f"{run_id} {run.state}" + (f": {run.error}" if run.error else ""))
        return run

    def stop(self, run_id: str, *, wait: bool = False, timeout: float | None = None, poll_interval: float = 10) -> RLRun:
        """Stop the run. Billing stops at once and the state becomes `closing`; the GPUs are released about a
        minute later, after the model is saved if it trained any step (then `jobs.wait_checkpoint(run_id)`).
        With `wait`, returns once it is `closed`."""
        run = RLRun.model_validate(self._client._post(f"/v1/training/sessions/{run_id}/close", retry=True))
        if wait:
            run = poll(lambda: self.get(run_id), lambda r: r.done, timeout, poll_interval)
        return run

    def metrics(self, run_id: str) -> builtins.list[RLStep]:
        """The record of every training call so far: reward statistics, where the time went (sampling, training,
        waiting for the client), how full the trainer's rows were. Also after the run has stopped."""
        return [RLStep.model_validate(x) for x in self._client._get(f"/v1/training/jobs/{run_id}/metrics")]

    def rollouts(self, run_id: str, *, step: int | None = None) -> RLRollouts:
        """What the model wrote in training call `step` (default: the latest): its best and its worst group with the
        rewards, kept for calls made through `connect`. Read a few as a run goes: a reward that rises while the
        replies get worse is the model gaming the reward."""
        query = "" if step is None else f"?step={step}"
        return RLRollouts.model_validate(self._client._get(f"/v1/training/sessions/{run_id}/rollouts{query}"))

    def connect(self, run_id: str, *, log_dir: str | Path | None = None, log_groups: int = 4) -> LiveRun:
        """Drive a ready run with NexCloud's own calls (`sample`, `train`): the short way, see `LiveRun`. For code
        written for the `tinker` SDK use `env` instead. `log_dir`: a directory for `metrics.jsonl` and
        `rollouts.jsonl` (what the model wrote in the first `log_groups` groups of every step, with the rewards)."""
        return LiveRun(self._client, run_id, log_dir=log_dir, log_groups=log_groups)

    def env(self, run_id: str) -> dict[str, str]:
        """What the `tinker` SDK reads to talk to this run: `os.environ.update(client.rl.env(id))` before
        `import tinker`. TINKER_API_KEY holds your API key: do not print it."""
        return {"TINKER_BASE_URL": f"{str(self._client._http.base_url).rstrip('/')}/v1/training/sessions/{run_id}/tinker",
                "TINKER_API_KEY": f"tml-{self._client._api_key}"}
