from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ._client import APIError, NexTronError
from .types import Episode, Sample, SampleBatch, TrainStep

if TYPE_CHECKING:
    from ._client import NexTron

Prompt = str | Sequence[dict[str, Any]] | dict[str, Any]
PAGE = 64  # prompts per answer when a batch's replies are fetched
POLL_WAIT = 30  # seconds the server holds one poll
OUTAGE_SECONDS = 120  # how long a call keeps trying through a restart of the API


class LiveRun:
    """A ready RL run driven with NexTron's own calls: sample a batch of prompts, score the replies in your own
    code, train on the rewards. NexTron keeps the sampled tokens and computes the advantages, so only your prompts and
    rewards travel. `client.rl.connect(run_id)` makes one; close it (or use `with`) when your loop ends.

    While it is open a background thread tells the run you are alive: a run whose client has been silent for
    3 minutes stops. One client drives a run: a run used this way refuses the `tinker` SDK, and the other way
    round.

    With `log_dir` every training call adds a line to two files there: `metrics.jsonl` (the step's record) and
    `rollouts.jsonl` (the first `log_groups` groups it trained: each episode's prompts, replies and reward), so you can
    read what the model wrote at any step."""

    def __init__(self, client: NexTron, run_id: str, *, heartbeat: float = 20.0, log_dir: str | Path | None = None,
                 log_groups: int = 4, base: str | None = None):
        self._client, self.id = client, run_id
        self._log_dir, self._log_groups = Path(log_dir) if log_dir else None, log_groups
        self._prompts: dict[str, dict[str, Any]] = {}  # "batch:index" -> the prompt, until the next training call
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        self._base = base or f"/v1/training/sessions/{run_id}/tinker/nextron/v1"   # an inference run's is under /v1/inference
        self._headers = {"X-API-Key": f"tml-{client._api_key}"}
        self._stop = threading.Event()
        self._call("POST", "/heartbeat")  # fails here if the run is not ready or is a tinker client's
        self._beat = threading.Thread(target=self._heartbeats, args=(heartbeat,), daemon=True)
        self._beat.start()

    def __enter__(self) -> LiveRun:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the heartbeat. The run itself goes on: `finish()` or `client.rl.stop(id)` ends it."""
        self._stop.set()

    def _heartbeats(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self._client._request("POST", f"{self._base}/heartbeat", headers=self._headers, retry=True)
            except (NexTronError, httpx.HTTPError):
                pass  # a missed beat is fine; three minutes of them stop the run

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        """One call to the run, repeated through a short outage: a restart of the API answers 500 or 502-504. Repeating
        is safe: a sample or train call carries its key."""
        deadline = time.monotonic() + OUTAGE_SECONDS
        while True:
            try:
                return self._client._request(method, self._base + path, json=body, headers=self._headers, retry=False).json()
            except (APIError, httpx.TransportError) as error:
                gone = isinstance(error, APIError) and error.status_code not in (500, 502, 503, 504)
                if gone or time.monotonic() > deadline:
                    raise
                time.sleep(2)

    def _wait(self, path: str, timeout: float | None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            answer: dict[str, Any] = self._call("GET", f"{path}?wait={POLL_WAIT}")
            if answer.get("error"):
                raise NexTronError(answer["error"])
            if answer["done"]:
                return answer
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"still not done after {timeout} s")

    def info(self) -> dict[str, Any]:
        """The run's model, its context length in tokens, and `round`: how many times its weights have changed."""
        answer: dict[str, Any] = self._call("GET", "/info")
        return answer

    def sample(self, prompts: Sequence[Prompt], *, n: int = 1, max_tokens: int | None = None, temperature: float = 1.0,
               top_p: float = 1.0, top_k: int = -1, stop: Sequence[str] | None = None,
               timeout: float | None = None) -> SampleBatch:
        """`n` replies to every prompt, sampled together from the current weights. A prompt is a string (one user
        message), a list of chat messages, or a dict: `{"messages": [...], "tools": [...]}` or `{"tokens": [ids]}`.
        `max_tokens` defaults to what the run's context leaves after the prompt. Returns the replies grouped by
        prompt, in order: `batch[i][j].text`."""
        started = time.monotonic()
        asked = [_prompt(p) for p in prompts]
        body = {"prompts": asked, "n": n, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "top_k": top_k, "stop": list(stop or []), "key": uuid.uuid4().hex}
        made = self._call("POST", "/sample", body)
        if self._log_dir:
            self._prompts.update({f"{made['batch']}:{index}": prompt for index, prompt in enumerate(asked)})
        path = f"/batches/{made['batch']}"
        done = self._wait(path, timeout)
        with ThreadPoolExecutor(8) as pool:  # the replies of a large batch are megabytes: fetch its pages side by side
            pages = pool.map(lambda start: self._call("GET", f"{path}?start={start}&count={PAGE}")["samples"],
                             range(0, made["prompts"], PAGE))
            samples: list[Any] = [group for page in pages for group in page]
        return SampleBatch(id=made["batch"], samples=samples, generate_seconds=done.get("generate_seconds"),
                           seconds=round(time.monotonic() - started, 3))

    def train(self, batch: Sequence[Sequence[Sample]] | None = None, rewards: Sequence[Sequence[float]] | None = None, *,
              learning_rate: float, episodes: Sequence[Episode] | None = None, advantage: str = "centered",
              loss: str = "importance_sampling", clip_low: float | None = None, clip_high: float | None = None,
              substeps: int = 1, beta1: float | None = None, beta2: float | None = None, eps: float | None = None,
              weight_decay: float | None = None, timeout: float | None = None) -> TrainStep:
        """Train on sampled replies and wait for the step. Either `batch` and `rewards` of the same shape (one reward
        per reply; a reply is compared with the other replies to its prompt), or `episodes`: each a list of replies in
        turn order (a conversation) with one reward and, optionally, the name of the group it is compared with.

        `advantage`: "centered" (reward minus its group's mean), "normalized" (also divided by the group's standard
        deviation) or "raw" (the reward is the advantage). `loss`: "importance_sampling", "ppo" or "cispo";
        `clip_low` / `clip_high` replace its clip window. `substeps` splits the episodes over that many optimizer
        steps. The weights change with this call: replies sampled before it cannot be trained afterwards."""
        started = time.monotonic()
        if episodes is None:
            if batch is None or rewards is None:
                raise ValueError("give batch and rewards, or episodes")
            if [len(group) for group in batch] != [len(group) for group in rewards]:
                raise ValueError("rewards must have the shape of the batch: one per reply")
            episodes = [Episode(samples=[s], reward=r) for group, scores in zip(batch, rewards) for s, r in zip(group, scores)]
        body: dict[str, Any] = {"episodes": [_episode(e) for e in episodes], "learning_rate": learning_rate,
                                "advantage": advantage, "loss": loss, "substeps": substeps, "key": uuid.uuid4().hex}
        optional = {"clip_low": clip_low, "clip_high": clip_high, "beta1": beta1, "beta2": beta2, "eps": eps,
                    "weight_decay": weight_decay}
        body.update({key: value for key, value in optional.items() if value is not None})
        made = self._call("POST", "/train", body)
        done = self._wait(f"/steps/{made['step']}", timeout)
        result = TrainStep.model_validate(done["metrics"] | {"seconds": round(time.monotonic() - started, 3)})
        if self._log_dir:
            self._log(self._log_dir, result, episodes)
        return result

    def _log(self, log_dir: Path, result: TrainStep, episodes: Sequence[Episode]) -> None:
        with open(log_dir / "metrics.jsonl", "a") as f:
            f.write(result.model_dump_json() + "\n")
        groups: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            turns = []
            for s in episode.samples:
                sample_id = s if isinstance(s, str) else s.id
                turn: dict[str, Any] = {"prompt": self._prompts.get(sample_id.rsplit(":", 1)[0])}
                if not isinstance(s, str):
                    turn.update(reply=s.text, finish_reason=s.finish_reason, tokens=s.tokens)
                turns.append(turn)
            first = episode.samples[0]
            group = episode.group or (first if isinstance(first, str) else first.id).rsplit(":", 1)[0]
            if group in groups or len(groups) < self._log_groups:
                groups.setdefault(group, []).append({"reward": episode.reward, "turns": turns})
        with open(log_dir / "rollouts.jsonl", "a") as f:
            for group, logged in groups.items():
                f.write(json.dumps({"step": result.step, "group": group, "episodes": logged}) + "\n")
        self._prompts.clear()

    def finish(self) -> None:
        """Your loop is over: the run saves the model if it trained and stops (billing stops)."""
        self.close()
        self._call("POST", "/finish")


def _prompt(prompt: Prompt) -> dict[str, Any]:
    if isinstance(prompt, str):
        return {"messages": [{"role": "user", "content": prompt}]}
    if isinstance(prompt, dict):
        return prompt
    return {"messages": list(prompt)}


def _episode(episode: Episode) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": [s if isinstance(s, str) else s.id for s in episode.samples], "reward": episode.reward}
    if episode.group is not None:
        out["group"] = episode.group
    return out
