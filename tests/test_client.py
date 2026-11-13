import json
from pathlib import Path

import httpx
import pytest

from nextron_cloud import APIError, NexTronError
from tests.conftest import body

JOB = {"id": "job_1", "state": "queued", "model": "qwen3.8-27b", "layout": "auto", "dataset_id": "ds_1",
       "params": {}, "created": 1.0, "updated": 1.0}


def test_create_job_retries_with_the_same_idempotency_key(make_client):
    keys = []

    def handler(request):
        keys.append(request.headers["Idempotency-Key"])
        if len(keys) == 1:
            return httpx.Response(503)
        return httpx.Response(201, json=JOB)

    client = make_client(handler)
    job = client.jobs.create(model="qwen3.8-27b", dataset_id="ds_1", params={"max_steps": 3})
    assert job.id == "job_1"
    assert len(keys) == 2 and keys[0] == keys[1]


def test_errors_carry_the_server_message(make_client):
    def handler(request):
        if request.url.path == "/v1/me":
            return httpx.Response(402, json={"error": "insufficient_balance", "detail": "this job needs about $3.00"})
        return httpx.Response(422, json={"detail": "the dataset has not passed its format check"})

    client = make_client(handler)
    with pytest.raises(APIError) as with_code:
        client.me()
    with pytest.raises(APIError) as plain:
        client.jobs.get("job_1")
    assert (with_code.value.status_code, with_code.value.code, with_code.value.message) == (402, "insufficient_balance", "this job needs about $3.00")
    assert plain.value.message == "the dataset has not passed its format check"


def test_upload_sends_parts_then_completes(make_client, tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"prompt": "1+1?", "completion": "2"}) + "\n")
    calls = []

    def handler(request):
        calls.append((request.method, str(request.url)))
        if request.url.path == "/v1/datasets":
            assert body(request) == {"name": "train.jsonl", "bytes": path.stat().st_size}
            return httpx.Response(201, json={"id": "ds_1", "part_bytes": 20, "part_urls": ["https://s3.test/1", "https://s3.test/2"]})
        if request.url.host == "s3.test":
            assert "Authorization" not in request.headers
            return httpx.Response(200)
        return httpx.Response(200, json={"id": "ds_1", "name": "train.jsonl", "bytes": 1, "state": "checking", "created": 1.0})

    client = make_client(handler)
    dataset = client.datasets.upload(path)
    assert dataset.state == "checking"
    assert [c[1] for c in calls] == ["https://api.test/v1/datasets", "https://s3.test/1", "https://s3.test/2",
                                     "https://api.test/v1/datasets/ds_1/complete"]


def test_upload_refuses_a_bad_file_before_sending_it(make_client, tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "no assistant turn"}) + "\n")
    client = make_client(lambda request: pytest.fail("nothing should be sent"))
    with pytest.raises(NexTronError, match="line 1: is not a format we know"):
        client.datasets.upload(path)


def test_logs_follow_pages_until_the_job_ends(make_client):
    pages = [[{"seq": 1, "ts": 1.0, "line": "a"}, {"seq": 2, "ts": 1.0, "line": "b"}], [], [{"seq": 3, "ts": 2.0, "line": "c"}], [], [{"seq": 4, "ts": 3.0, "line": "d"}], []]
    states = ["running", "succeeded"]

    def handler(request):
        if request.url.path.endswith("/logs"):
            lines = pages.pop(0)
            return httpx.Response(200, json={"lines": lines, "next_cursor": lines[-1]["seq"] if lines else int(request.url.params["cursor"])})
        return httpx.Response(200, json=JOB | {"state": states.pop(0)})

    client = make_client(handler)
    assert [line.line for line in client.jobs.logs("job_1", follow=True)] == ["a", "b", "c", "d"]


def test_download_checkpoint_resumes_a_partial_file_and_checks_it(make_client, tmp_path: Path):
    import hashlib
    data = b"0123456789"
    (tmp_path / "model.safetensors").write_bytes(data[:4])

    def handler(request):
        if request.url.host == "r2.test":
            start = int(request.headers["Range"].removeprefix("bytes=").removesuffix("-"))
            return httpx.Response(206, content=data[start:])
        files = [{"name": n, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": "https://r2.test/m"}
                 for n in ("model.safetensors", "other.safetensors", "config.json")]
        return httpx.Response(200, json={"step": 3, "bytes": len(data), "expires": 9.0, "files": files})

    client = make_client(handler)
    client.jobs.download_checkpoint("job_1", tmp_path, include=["model.*"])
    assert (tmp_path / "model.safetensors").read_bytes() == data
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.safetensors"]


def test_cli_create_without_yes_only_estimates(make_client, monkeypatch, capsys):
    from nextron_cloud import cli
    posted = []

    def handler(request):
        posted.append((request.url.path, body(request)["params"]))
        return httpx.Response(200, json={"metered": True, "tokens": 1, "price_per_mtok": 100, "credits": 1, "balance": 5, "enough": True})

    make_client(handler)
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    monkeypatch.setenv("NEXTRON_BASE_URL", "https://api.test")
    with pytest.raises(SystemExit) as stop:
        cli.main(["jobs", "create", "--model", "qwen3.8-27b", "--dataset", "ds_1", "--param", "max_steps=3", "--param", "train_type=lora"])
    assert stop.value.code == 3
    assert posted == [("/v1/training/estimate", {"max_steps": 3, "train_type": "lora"})]
    assert json.loads(capsys.readouterr().out)["estimate"]["credits"] == 1


def test_training_and_checkpoint_are_waited_for_apart(make_client):
    running = JOB | {"state": "running"}
    uploading = JOB | {"state": "succeeded", "checkpoint": {"status": "uploading"}}
    ready = JOB | {"state": "succeeded", "checkpoint": {"status": "ready"}}
    answers = [running, uploading, uploading, ready]
    client = make_client(lambda request: httpx.Response(200, json=answers.pop(0)))
    assert client.jobs.wait("job_1").state == "succeeded"
    assert client.jobs.wait_checkpoint("job_1").checkpoint_ready


def test_a_sweep_started_twice_creates_each_job_once(make_client):
    from nextron_cloud import experiments
    created = {}

    def handler(request):
        sent = body(request)
        key = request.headers["Idempotency-Key"]
        created.setdefault(key, (sent["name"], sent["tags"], sent["params"]))
        return httpx.Response(201, json=JOB | {"id": f"job_{key[:6]}"})

    client = make_client(handler)
    run = lambda: experiments.sweep(client, name="lr try", model="qwen3.8-27b", dataset_id="ds_1",
                                    axes={"learning_rate": [1e-5, 2e-5]}, params={"max_steps": 3}, start=True)
    first, second = run(), run()
    assert [r["job_id"] for r in first] == [r["job_id"] for r in second]
    assert sorted(created.values()) == [
        ("lr try learning_rate=1e-05", ["sweep:lr-try"], {"max_steps": 3, "learning_rate": 1e-5}),
        ("lr try learning_rate=2e-05", ["sweep:lr-try"], {"max_steps": 3, "learning_rate": 2e-5}),
    ]


def test_estimate_raises_with_the_reason_the_server_would_refuse(make_client):
    refused = {"metered": True, "tokens": 1, "price_per_mtok": 135, "credits": 1, "balance": 5, "enough": True,
               "valid": False, "error": "lora training of mimo-311b is coming soon"}
    client = make_client(lambda request: httpx.Response(200, json=refused))
    with pytest.raises(APIError, match="coming soon") as error:
        client.jobs.estimate(model="mimo-311b", dataset_id="ds_1", params={"train_type": "lora"})
    assert error.value.status_code == 422


def test_events_follow_until_the_checkpoint_when_asked(make_client):
    pages = [[{"seq": 1, "ts": 1.0, "type": "job.state", "data": {"state": "succeeded"}}], [],
             [{"seq": 2, "ts": 2.0, "type": "checkpoint.stage", "data": {"stage": "ready"}}], [], []]
    jobs = [JOB | {"state": "succeeded", "checkpoint": {"status": "uploading"}}, JOB | {"state": "succeeded", "checkpoint": {"status": "ready"}}]

    def handler(request):
        if request.url.path.endswith("/events"):
            events = pages.pop(0)
            return httpx.Response(200, json={"events": events, "next_cursor": events[-1]["seq"] if events else 0})
        return httpx.Response(200, json=jobs.pop(0))

    client = make_client(handler)
    assert [e.seq for e in client.jobs.events("job_1", follow=True, until="checkpoint")] == [1, 2]


def test_the_key_is_never_sent_over_plain_http():
    from nextron_cloud import NexTron
    with pytest.raises(NexTronError, match="https"):
        NexTron("nk_test", base_url="http://nextron.example.com")


def test_a_checkpoint_file_name_cannot_leave_the_destination(make_client, tmp_path: Path):
    files = [{"name": "../../.bashrc", "bytes": 1, "sha256": "0" * 64, "url": "https://r2.test/x"}]
    client = make_client(lambda request: httpx.Response(200, json={"step": 1, "bytes": 1, "expires": 9.0, "files": files}))
    with pytest.raises(NexTronError, match="not a plain file name"):
        client.jobs.download_checkpoint("job_1", tmp_path / "ckpt")
    assert not (tmp_path / ".bashrc").exists()


def test_cli_sample_table_counts_samples_over_the_models_longest(make_client, monkeypatch, capsys):
    from nextron_cloud import cli
    stats = {"total": 900, "max": 70000, "over": {"32768": 5, "65536": 2}}
    datasets = [{"id": "ds_s", "name": "Math", "bytes": 1, "state": "ready", "created": 1.0, "samples": 10, "sample": True,
                 "tokens": {"m": stats}},
                {"id": "ds_own", "name": "mine", "bytes": 1, "state": "ready", "created": 1.0, "sample": False}]
    method = {"max_seq_len": 65536, "layouts": []}
    model = {"id": "m", "name": "M", "maker": "x", "params": "1B", "active_params": "1B", "architecture": "x",
             "input": ["text"], "methods": {"full": method}, "batch_size": True, "validation_set": True, "layouts": {}}
    answers = {"/v1/datasets": datasets, "/v1/models": [model]}
    make_client(lambda request: httpx.Response(200, json=answers[request.url.path]))
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    monkeypatch.setenv("NEXTRON_BASE_URL", "https://api.test")
    cli.main(["datasets", "list", "--sample", "--table", "--model", "m"])
    header, row = capsys.readouterr().out.splitlines()
    assert header.split("\t")[-1] == "over_65536"
    assert row.split("\t") == ["ds_s", "Math", "ready", "10", "False", "900", "70000", "2"]
