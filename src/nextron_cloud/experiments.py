"""Experiments over many jobs: a sweep (one job per combination of settings) and a report comparing jobs."""
from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from .types import Estimate, Job, TrainingParams

if TYPE_CHECKING:
    from ._client import NexTron


def grid(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of the axes' values: {"learning_rate": [1, 2], "num_train_epochs": [3]} -> 2 dicts."""
    combos: list[dict[str, Any]] = [{}]
    for key, values in axes.items():
        combos = [dict(combo, **{key: value}) for combo in combos for value in values]
    return combos


def sweep_tag(name: str) -> str:
    """The tag every job of the sweep `name` carries."""
    return "sweep:" + re.sub(r"[^A-Za-z0-9_.:=/+-]+", "-", name)[:58]


def sweep(client: NexTron, *, name: str, model: str, dataset_id: str, axes: dict[str, list[Any]],
          params: TrainingParams | None = None, layout: str = "auto", start: bool = False) -> list[dict[str, Any]]:
    """One job per combination of `axes` on top of `params`. Without `start`, only each job's estimate. With it, the
    jobs are created, named "<name> key=value ..." and tagged `sweep_tag(name)`; running it again creates nothing new
    (each job's idempotency key comes from the sweep and its settings)."""
    rows = []
    for combo in grid(axes):
        job_params: TrainingParams = {**(params or {}), **combo}  # type: ignore[typeddict-item]
        label = " ".join([name] + [f"{key}={value}" for key, value in combo.items()])
        draft = {"model": model, "dataset_id": dataset_id, "layout": layout, "params": job_params}
        if not start:
            estimate: Estimate = client.jobs.estimate(**draft)  # type: ignore[arg-type]
            rows.append({"name": label, "params": job_params, "estimate": estimate.model_dump(mode="json")})
            continue
        key = hashlib.sha256(json.dumps([name, draft], sort_keys=True).encode()).hexdigest()[:32]
        job = client.jobs.create(**draft, name=label, tags=[sweep_tag(name)], idempotency_key=key)  # type: ignore[arg-type]
        rows.append({"name": label, "params": job_params, "job_id": job.id, "state": job.state})
    return rows


def summarize(client: NexTron, job: Job) -> dict[str, Any]:
    """One job as a report row: its settings, losses, tokens, training time and checkpoint."""
    job = client.jobs.get(job.id)
    points = client.jobs.metrics(job.id) if job.state != "queued" else []
    evals = [p for p in points if p.val_loss is not None]
    best = min(evals, key=lambda p: p.val_loss or 0.0) if evals else None
    timeline = job.timeline or {}
    start, end = timeline.get("training"), timeline.get("trained")
    return {
        "id": job.id, "name": job.name, "tags": job.tags, "state": job.state, "model": job.model, "layout": job.layout,
        "train_type": job.params.get("train_type"), "learning_rate": job.params.get("learning_rate"),
        "num_train_epochs": job.params.get("num_train_epochs"), "max_steps": job.params.get("max_steps"),
        "steps": points[-1].step if points else None, "loss": points[-1].loss if points else None,
        "val_loss": evals[-1].val_loss if evals else None,
        "best_val_loss": best.val_loss if best else None, "best_val_step": best.step if best else None,
        "tokens_trained": job.tokens_trained, "training_minutes": round((end - start) / 60, 1) if start and end else None,
        "charged_credits": client.jobs.billing(job.id).charged_credits if job.done else None,
        "checkpoint_ready": job.checkpoint_ready, "error": job.error,
    }


def report(client: NexTron, *, tag: str | None = None, job_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Report rows for the jobs carrying `tag`, or the given ids, oldest first."""
    if job_ids:
        jobs = [client.jobs.get(job_id) for job_id in job_ids]
    else:
        jobs = list(reversed(client.jobs.list(tag=tag)))
    return [summarize(client, job) for job in jobs]
