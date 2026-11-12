from __future__ import annotations

import builtins
import fnmatch
import hashlib
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ._client import APIError, NexTronError, poll
from .types import Billing, Checkpoint, Estimate, Event, Job, LogLine, MetricPoint, Progress, TrainingParams, Wandb

if TYPE_CHECKING:
    from ._client import NexTron


class Jobs:
    """Training jobs: one request trains a model on a dataset and leaves a checkpoint to download."""

    def __init__(self, client: NexTron):
        self._client = client

    def create(self, *, model: str, dataset_id: str, layout: str = "auto", params: TrainingParams | None = None,
               name: str | None = None, tags: builtins.list[str] | None = None, wandb: Wandb | None = None,
               idempotency_key: str | None = None) -> Job:
        """Queue a job. `name` and `tags` are your labels (tags group runs, e.g. one sweep). Sending the same
        `idempotency_key` again returns the same job instead of a second one, so the request is retried safely; one is
        made up when none is given."""
        body = _draft(model, dataset_id, layout, params)
        if name:
            body["name"] = name
        if tags:
            body["tags"] = tags
        if wandb:
            body["wandb"] = wandb
        headers = {"Idempotency-Key": idempotency_key or uuid.uuid4().hex}
        return Job.model_validate(self._client._post("/v1/training/jobs", body, headers=headers, retry=True))

    def estimate(self, *, model: str, dataset_id: str, layout: str = "auto", params: TrainingParams | None = None) -> Estimate:
        """What the job would cost, the settings and steps it would run with, and warnings; free. Raises APIError (422) when
        the server would refuse the job, with its reason."""
        body = _draft(model, dataset_id, layout, params)
        estimate = Estimate.model_validate(self._client._post("/v1/training/estimate", body, retry=True))
        if not estimate.valid:
            raise APIError(422, str(estimate.error), "invalid_job", estimate.model_dump())
        return estimate

    def list(self, *, tag: str | None = None) -> list[Job]:
        """Every job of the organisation, newest first; with `tag`, only those carrying it."""
        jobs = [Job.model_validate(x) for x in self._client._get("/v1/training/jobs")]
        return [j for j in jobs if tag in j.tags] if tag else jobs

    def get(self, job_id: str) -> Job:
        return Job.model_validate(self._client._get(f"/v1/training/jobs/{job_id}"))

    def cancel(self, job_id: str) -> Job:
        """Cancel a queued job now, or ask a running one to stop."""
        return Job.model_validate(self._client._post(f"/v1/training/jobs/{job_id}/cancel", retry=True))

    def wait(self, job_id: str, *, timeout: float | None = None, poll_interval: float = 30) -> Job:
        """Wait until training is over: succeeded, failed or cancelled. A failed job is returned, not raised. The
        checkpoint uploads after that; see `wait_checkpoint`."""
        return poll(lambda: self.get(job_id), lambda j: j.done, timeout, poll_interval)

    def wait_checkpoint(self, job_id: str, *, timeout: float | None = None, poll_interval: float = 60) -> Job:
        """Wait until a succeeded job's checkpoint is uploaded and can be downloaded (minutes to an hour after training,
        by model size). NexTronError if the job ended without one."""
        job = poll(lambda: self.get(job_id), lambda j: j.checkpoint_ready or (j.done and j.state != "succeeded"),
                   timeout, poll_interval)
        if not job.checkpoint_ready:
            raise NexTronError(f"{job_id} {job.state}; only a succeeded job has a checkpoint")
        return job

    def progress(self, job_id: str) -> Progress:
        """Where the job is now: phase, queue ahead, training ETA, checkpoint upload bytes, last activity."""
        return Progress.model_validate(self._client._get(f"/v1/training/jobs/{job_id}/progress"))

    def events(self, job_id: str, *, after: int = 0, follow: bool = False, until: str = "training",
               poll_interval: float = 15) -> Iterator[Event]:
        """The job's events in order from `after`. With `follow`, keep reading until training is over (`until="training"`)
        or the checkpoint is uploaded too (`until="checkpoint"`)."""
        ended = False
        while True:
            page = self._client._get(f"/v1/training/jobs/{job_id}/events", params={"after": after, "limit": 5000})
            for event in page["events"]:
                yield Event.model_validate(event)
            after = page["next_cursor"]
            if page["events"]:
                continue
            if not follow or ended:
                return
            job = self.get(job_id)
            ended = job.done and (until == "training" or job.state != "succeeded" or job.checkpoint_ready)
            if not ended:
                time.sleep(poll_interval)

    def billing(self, job_id: str) -> Billing:
        """What the job holds and was charged (US cents)."""
        return Billing.model_validate(self._client._get(f"/v1/training/jobs/{job_id}/billing"))

    def metrics(self, job_id: str) -> builtins.list[MetricPoint]:
        """Loss, learning rate and throughput of every step so far."""
        return [MetricPoint.model_validate(x) for x in self._client._get(f"/v1/training/jobs/{job_id}/metrics")]

    def logs(self, job_id: str, *, follow: bool = False, poll_interval: float = 10) -> Iterator[LogLine]:
        """The job's log lines from the start; with `follow`, keep reading until the job has ended."""
        cursor, ended = 0, False
        while True:
            page = self._client._get(f"/v1/training/jobs/{job_id}/logs", params={"cursor": cursor, "limit": 5000})
            for line in page["lines"]:
                yield LogLine.model_validate(line)
            cursor = page["next_cursor"]
            if page["lines"]:
                continue
            if not follow or ended:
                return
            ended = self.get(job_id).done  # once ended, one more read picks up the last lines
            if not ended:
                time.sleep(poll_interval)

    def checkpoint(self, job_id: str) -> Checkpoint:
        """The trained model's files with 24-hour links. Checkpoints are kept 30 days after training."""
        return Checkpoint.model_validate(self._client._get(f"/v1/training/jobs/{job_id}/checkpoint"))

    def download_checkpoint(self, job_id: str, dest: str | os.PathLike[str], *, include: builtins.list[str] | None = None) -> Path:
        """Download the checkpoint into `dest` and check every file's sha256. `include`: only the files matching one of
        these glob patterns (e.g. ["*.json", "*.jinja"]). A file already complete is skipped, a partial one resumed."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        files = self.checkpoint(job_id).files
        if include is not None:
            files = [f for f in files if any(fnmatch.fnmatch(f.name, pattern) for pattern in include)]
            if not files:
                raise NexTronError(f"no checkpoint file matches {include}")
        with httpx.Client(timeout=600, follow_redirects=True) as storage:
            for f in files:
                if Path(f.name).name != f.name or f.name in (".", ".."):
                    raise NexTronError(f"refusing checkpoint file name {f.name!r}: not a plain file name")
                path = dest / f.name
                _download(storage, f.url, path, f.bytes)
                if _sha256(path) != f.sha256:
                    path.unlink()
                    raise IOError(f"{f.name}: sha256 does not match the checkpoint's; removed it, run again to download it anew")
        return dest

    def export(self, job_id: str, dest: str | os.PathLike[str]) -> Path:
        """Save the job's record as a zip: config, logs, metrics, GPU telemetry and the manifest."""
        dest = Path(dest)
        dest.write_bytes(self._client._request("GET", f"/v1/training/jobs/{job_id}/export").content)
        return dest

    def manifest(self, job_id: str) -> bytes:
        """The manifest exactly as stored; its sha256 is the job's `manifest_sha256`."""
        return self._client._request("GET", f"/v1/training/jobs/{job_id}/manifest").content

    def telemetry(self, job_id: str) -> dict[str, Any]:
        """The hardware the job ran on and GPU/host utilisation while it trained."""
        return dict(self._client._get(f"/v1/training/jobs/{job_id}/telemetry"))

    def trace(self, job_id: str) -> bytes:
        """The engine's per-rank step timeline, a Chrome trace for ui.perfetto.dev."""
        return self._client._request("GET", f"/v1/training/jobs/{job_id}/trace").content


def _draft(model: str, dataset_id: str, layout: str, params: TrainingParams | None) -> dict[str, Any]:
    return {"model": model, "dataset_id": dataset_id, "layout": layout, "params": dict(params or {})}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(storage: httpx.Client, url: str, path: Path, size: int) -> None:
    have = path.stat().st_size if path.exists() else 0
    if have == size:
        return
    if have > size:
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with storage.stream("GET", url, headers=headers) as r:
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with path.open(mode) as out:
            for chunk in r.iter_bytes(1 << 20):
                out.write(chunk)
    if path.stat().st_size != size:
        raise IOError(f"{path.name}: got {path.stat().st_size} bytes, expected {size}; run again to resume")
