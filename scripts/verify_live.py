"""Run every read, an estimate, and a small dataset upload/check/delete against a live deployment. Starts no job.

    NEXTRON_API_KEY=nk_live_... python scripts/verify_live.py
"""
import tempfile
from pathlib import Path

from nextron_cloud import NexTron, write_jsonl

client = NexTron()
me = client.me()
print("account", me.org.name, "balance $", me.org.credits / 100)
print("models", [(m.id, sorted(m.methods)) for m in client.models()])
print("layouts", [(layout.id, layout.available) for layout in client.layouts()])
datasets = client.datasets.list()
sample = next(d for d in datasets if d.sample and d.state == "ready")
print("datasets", len(datasets), "sample", sample.id, sample.name)
print("estimate", client.jobs.estimate(model="qwen3.8-27b", dataset_id=sample.id, params={"max_steps": 10}))

jobs = client.jobs.list()
done = next(j for j in jobs if j.checkpoint_ready)
print("jobs", len(jobs), "latest succeeded", done.id, done.model)
print("metrics", len(client.jobs.metrics(done.id)), "steps")
print("progress", client.jobs.progress(done.id).phase, "events", sum(1 for _ in client.jobs.events(done.id)),
      "billing", client.jobs.billing(done.id).billing)
print("logs", sum(1 for _ in client.jobs.logs(done.id)), "lines")
print("telemetry", list(client.jobs.telemetry(done.id)))
print("checkpoint files", len(client.jobs.checkpoint(done.id).files))
print("manifest", len(client.jobs.manifest(done.id)), "bytes")
with tempfile.TemporaryDirectory() as tmp:
    print("export", client.jobs.export(done.id, Path(tmp) / "export.zip").stat().st_size, "bytes")
    for samples in (50, 200_000):   # one PUT, and a multipart upload (over 16 MB)
        path = Path(tmp) / f"sdk-live-check-{samples}.jsonl"
        write_jsonl(({"question": f"What is {i} plus {i}?", "answer": str(2 * i)} for i in range(samples)), path)
        checked = client.datasets.wait(client.datasets.upload(path).id, timeout=1800)
        print("upload", path.stat().st_size, "bytes", checked.id, checked.state, checked.samples, "samples", checked.errors)
        print("download_url", client.datasets.download_url(checked.id)[:60], "...")
        client.datasets.delete(checked.id)
