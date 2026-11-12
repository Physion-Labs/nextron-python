from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ._client import NexTronError, poll
from .data import check_file
from .types import Dataset

if TYPE_CHECKING:
    from ._client import NexTron


class Datasets:
    """Training data: JSONL files you upload or import from Hugging Face, checked by the server before use."""

    def __init__(self, client: NexTron):
        self._client = client

    def list(self) -> list[Dataset]:
        """Your datasets and the sample datasets every account can use."""
        return [Dataset.model_validate(x) for x in self._client._get("/v1/datasets")]

    def get(self, dataset_id: str) -> Dataset:
        return Dataset.model_validate(self._client._get(f"/v1/datasets/{dataset_id}"))

    def delete(self, dataset_id: str) -> None:
        self._client._request("DELETE", f"/v1/datasets/{dataset_id}")

    def upload(self, path: str | os.PathLike[str], *, name: str | None = None, check: bool = True) -> Dataset:
        """Upload a JSONL file. With `check`, the server's format check runs locally first, so a bad file fails before
        it is sent. The dataset is "checking" when this returns; `wait` for "ready"."""
        path = Path(path)
        if check:
            result = check_file(path)
            if not result.ok:
                raise NexTronError(f"{path.name} would not pass the format check: " + "; ".join(result.errors))
        size = path.stat().st_size
        created = self._client._post("/v1/datasets", {"name": name or path.name, "bytes": size})
        with httpx.Client(timeout=600) as storage, path.open("rb") as f:
            if "part_urls" in created:
                for url in created["part_urls"]:
                    _put(storage, url, f.read(created["part_bytes"]), {})
            else:
                _put(storage, created["upload_url"], f.read(), created["upload_headers"])
        return Dataset.model_validate(self._client._post(f"/v1/datasets/{created['id']}/complete"))

    def huggingface_splits(self, url: str, *, hf_token: str | None = None) -> builtins.list[dict[str, Any]]:
        """The configs and splits of a Hugging Face dataset, with their rows and sizes."""
        body = {"url": url, "hf_token": hf_token}
        return list(self._client._post("/v1/datasets/huggingface/splits", body)["splits"])

    def import_huggingface(self, url: str, *, config: str = "default", split: str = "train", name: str | None = None,
                           hf_token: str | None = None) -> Dataset:
        """Import one split of a Hugging Face dataset; `wait` for "ready"."""
        body = {"url": url, "config": config, "split": split, "name": name, "hf_token": hf_token}
        return Dataset.model_validate(self._client._post("/v1/datasets/huggingface", body))

    def download_url(self, dataset_id: str) -> str:
        """A one-hour link to the dataset as training reads it (after any format conversion)."""
        return str(self._client._get(f"/v1/datasets/{dataset_id}/download")["url"])

    def wait(self, dataset_id: str, *, timeout: float | None = 3600, poll_interval: float = 5) -> Dataset:
        """Wait until the format check ends: "ready", or "invalid" with `errors` saying why."""
        return poll(lambda: self.get(dataset_id), lambda d: d.done, timeout, poll_interval)


def _put(storage: httpx.Client, url: str, data: bytes, headers: dict[str, str], attempts: int = 4) -> None:
    for attempt in range(attempts):
        try:
            storage.put(url, content=data, headers=headers).raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == attempts - 1:
                raise
