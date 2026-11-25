from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from ._client import NexTronError
from .datasets import _put
from .types import UploadedCheckpoint

if TYPE_CHECKING:
    from ._client import NexTron

PARTS_AT_ONCE = 8


class Checkpoints:
    """Model weights you upload to start jobs from: `from_job` takes an uploaded checkpoint's id as it takes a job's."""

    def __init__(self, client: NexTron):
        self._client = client

    def list(self) -> list[UploadedCheckpoint]:
        return [UploadedCheckpoint.model_validate(x) for x in self._client._get("/v1/checkpoints")]

    def get(self, checkpoint_id: str) -> UploadedCheckpoint:
        return UploadedCheckpoint.model_validate(self._client._get(f"/v1/checkpoints/{checkpoint_id}"))

    def delete(self, checkpoint_id: str) -> None:
        self._client._request("DELETE", f"/v1/checkpoints/{checkpoint_id}")

    def upload(self, directory: str | os.PathLike[str], *, model: str, name: str | None = None) -> UploadedCheckpoint:
        """Upload the weights of `model` in `directory`: its *.safetensors files, float32 or bfloat16, with the base
        model's tensor names and shapes. Only the weights are sent; config and tokenizer stay the base model's.
        Returns the checkpoint "ready"; NexTronError with the server's reasons when the files are not that model."""
        files = sorted(Path(directory).glob("*.safetensors"))
        if not files:
            raise NexTronError(f"no .safetensors files in {directory}")
        body = {"model": model, "name": name or Path(directory).resolve().name,
                "files": [{"name": f.name, "bytes": f.stat().st_size} for f in files]}
        created = self._client._post("/v1/checkpoints", body)
        size = created["part_bytes"]
        with httpx.Client(timeout=600) as storage, ThreadPoolExecutor(PARTS_AT_ONCE) as pool:
            for file in files:
                parts = range(1, -(-file.stat().st_size // size) + 1)
                for first in range(0, len(parts), PARTS_AT_ONCE):
                    batch = list(parts[first:first + PARTS_AT_ONCE])
                    urls = self._client._post(f"/v1/checkpoints/{created['id']}/parts", {"name": file.name, "parts": batch}, retry=True)["urls"]
                    list(pool.map(lambda sent: _put(storage, sent[1], _part(file, sent[0], size), {}), zip(batch, urls)))
        checkpoint = UploadedCheckpoint.model_validate(self._client._post(f"/v1/checkpoints/{created['id']}/complete", retry=True))
        if checkpoint.state != "ready":
            raise NexTronError(f"{directory} is not a checkpoint of {model}: " + "; ".join(checkpoint.errors))
        return checkpoint


def _part(file: Path, number: int, size: int) -> bytes:
    with file.open("rb") as f:
        f.seek((number - 1) * size)
        return f.read(size)
