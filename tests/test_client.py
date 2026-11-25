import json
import time
from pathlib import Path

import httpx
import pytest

from nextron_cloud import APIError, NexTronError
from tests.conftest import body, mp4

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


def test_upload_packs_no_bundle_for_a_file_without_images(make_client, monkeypatch, tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"prompt": "1+1?", "completion": "2"}) + "\n")
    monkeypatch.setattr("nextron_cloud.datasets.write_bundle", lambda *args: pytest.fail("a text file is sent as it is"))

    def handler(request):
        if request.url.path == "/v1/datasets":
            return httpx.Response(201, json={"id": "ds_1", "upload_url": "https://s3.test/1", "upload_headers": {}})
        return httpx.Response(200, json={"id": "ds_1", "name": "train.jsonl", "bytes": 1, "state": "checking", "created": 1.0})

    assert make_client(handler).datasets.upload(path).state == "checking"


def test_upload_refuses_a_bad_file_before_sending_it(make_client, tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "no assistant turn"}) + "\n")
    client = make_client(lambda request: pytest.fail("nothing should be sent"))
    with pytest.raises(NexTronError, match="line 1: is not a format we know"):
        client.datasets.upload(path)


def test_a_checkpoint_upload_sends_each_files_parts_and_raises_what_the_check_found(make_client, tmp_path: Path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 25)
    (tmp_path / "config.json").write_text("{}")   # only the weights are sent
    sent, checkpoint = {}, {"id": "ckpt_1", "model": "qwen3.8-27b", "name": "mine", "state": "ready", "bytes": 25, "created": 1.0}

    def handler(request):
        if request.url.path == "/v1/checkpoints":
            assert body(request) == {"model": "qwen3.8-27b", "name": tmp_path.name, "files": [{"name": "a.safetensors", "bytes": 25}]}
            return httpx.Response(201, json=checkpoint | {"state": "uploading", "part_bytes": 10})
        if request.url.path.endswith("/parts"):
            assert body(request) == {"name": "a.safetensors", "parts": [1, 2, 3]}
            return httpx.Response(200, json={"urls": [f"https://r2.test/{n}" for n in (1, 2, 3)]})
        if request.url.host == "r2.test":
            sent[request.url.path] = len(request.content)
            return httpx.Response(200)
        return httpx.Response(200, json=checkpoint)

    client = make_client(handler)
    assert client.checkpoints.upload(tmp_path, model="qwen3.8-27b").id == "ckpt_1"
    assert sent == {"/1": 10, "/2": 10, "/3": 5}
    checkpoint |= {"state": "invalid", "errors": ["3 tensors are not the model's: a, b, c"]}
    with pytest.raises(NexTronError, match="not a checkpoint of qwen3.8-27b: 3 tensors"):
        client.checkpoints.upload(tmp_path, model="qwen3.8-27b")


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


def test_a_run_that_saved_no_checkpoint_ends_the_wait_for_one(make_client):
    saved_nothing = JOB | {"id": "rl_1", "state": "succeeded"}   # an RL run that trained no step
    polls = []

    def handler(request):
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"events": [], "next_cursor": 0})
        polls.append(request)
        if len(polls) > 3:
            pytest.fail("still waiting for a checkpoint that will not come")
        return httpx.Response(200, json=saved_nothing)

    client = make_client(handler)
    with pytest.raises(NexTronError, match="saved no checkpoint"):
        client.jobs.wait_checkpoint("rl_1")
    assert list(client.jobs.events("rl_1", follow=True, until="checkpoint")) == []


def test_a_deleted_checkpoint_ends_the_wait_for_one(make_client):
    deleted = JOB | {"state": "succeeded", "checkpoint": {"status": "deleted"}}
    client = make_client(lambda request: httpx.Response(200, json=deleted))
    with pytest.raises(NexTronError, match="was deleted"):
        client.jobs.wait_checkpoint("job_1", poll_interval=0)


def test_a_wait_runs_for_its_whole_timeout_before_it_gives_up(make_client):
    answers = [JOB | {"state": "running"}, JOB | {"state": "succeeded"}]
    client = make_client(lambda request: httpx.Response(200, json=answers.pop(0)))
    assert client.jobs.wait("job_1", timeout=0.2, poll_interval=30).state == "succeeded"


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


def test_serve_rewards_scores_each_batch_until_the_job_is_done(make_client):
    item = {"row": 0, "completion": "4", "answer": "4"}
    prompts = {"0": [{"role": "user", "content": "2+2"}]}
    pages = [{"state": "running", "batches": [{"id": "round00000_rank0", "created": 1.0, "prompts": prompts,
                                               "items": [item, item | {"completion": "5"}]}]},
             {"state": "running", "batches": []}]
    posted = []

    def handler(request):
        if request.url.path.endswith("/rollouts"):
            return httpx.Response(200, json=pages.pop(0) if pages else {"state": "succeeded", "batches": []})
        if request.url.path.endswith("/scores"):
            posted.append((request.url.path.rsplit("/", 2)[-2], body(request)["scores"]))
            return httpx.Response(200, json={"scored": 2})
        return httpx.Response(200, json={"id": "job_1", "state": "succeeded", "model": "m", "layout": "1x8-h100", "dataset_id": "ds",
                                         "params": {}, "created": 1.0, "updated": 1.0})

    client = make_client(handler)
    assert client.jobs.serve_rewards("job_1", lambda i: i["completion"] == i["answer"] and i["messages"] == prompts["0"]) == 1
    assert posted == [("round00000_rank0", [1.0, 0.0])]


def test_a_model_without_rl_runs_is_refused_a_quote(make_client, monkeypatch):
    nothing = {"price_per_gpu_hour": None, "gpus": None, "price_per_hour": None, "credits": None, "balance": 10000, "enough": True}
    make_client(lambda request: httpx.Response(200, json=nothing))
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    monkeypatch.setenv("NEXTRON_BASE_URL", "https://api.test")
    from nextron_cloud.cli import main
    with pytest.raises(SystemExit, match="has no Serverless RL"):
        main(["rl", "start", "--model", "nemotron-3-super-120b", "--max-hours", "1"])


def test_an_rl_run_is_priced_first_and_its_env_points_tinker_at_it_without_printing_the_key(make_client, monkeypatch, capsys):
    run = {"id": "rl_1", "state": "starting", "model": "qwen3.8-27b", "params": {"max_hours": 2}, "created": 1.0, "updated": 1.0}
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/estimate"):
            return httpx.Response(200, json={"price_per_gpu_hour": 300, "gpus": 8, "price_per_hour": 2400, "credits": 4800,
                                             "balance": 10000, "enough": True})
        if request.method == "POST":
            assert body(request) == {"model": "qwen3.8-27b", "max_hours": 2.0, "max_context": 8192, "grad_clip": False,
                                     "images": True}
            return httpx.Response(201, json=run)
        return httpx.Response(200, json=run | {"state": "ready", "ready": 5.0})

    from nextron_cloud import NexTron
    from nextron_cloud.cli import main
    make_client(handler)   # every client made from here on talks to the handler
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    monkeypatch.setenv("NEXTRON_BASE_URL", "https://api.test")
    create = ["rl", "start", "--model", "qwen3.8-27b", "--max-hours", "2", "--max-context", "8192"]
    with pytest.raises(SystemExit) as stop:
        main(create)
    assert stop.value.code == 3 and ("POST", "/v1/training/sessions") not in calls   # only the price without --yes
    printed = json.loads(capsys.readouterr().out)
    assert printed["price"] == "$3.00 per H100-hour (8 GPUs)" and printed["estimate"]["credits"] == 4800
    main(create + ["--yes", "--no-grad-clip", "--images"])
    assert ("POST", "/v1/training/sessions") in calls
    client = NexTron()
    assert client.rl.wait("rl_1").state == "ready"
    assert client.rl.env("rl_1") == {"TINKER_BASE_URL": "https://api.test/v1/training/sessions/rl_1/tinker",
                                            "TINKER_API_KEY": "tml-nk_test"}
    capsys.readouterr()
    main(["rl", "env", "rl_1"])
    printed = capsys.readouterr().out
    assert "nk_test" not in printed and 'TINKER_API_KEY="tml-$NEXTRON_API_KEY"' in printed


def test_stopping_an_rl_run_can_wait_until_the_gpus_are_released(make_client):
    states = iter(["closing", "closing", "closed"])

    def handler(request: httpx.Request) -> httpx.Response:
        state = "closing" if request.method == "POST" else next(states)
        return httpx.Response(200, json={"id": "rl_1", "state": state, "model": "m", "params": {}, "created": 1.0, "updated": 1.0,
                                         "billable_seconds": 600})

    client = make_client(handler)
    assert client.rl.stop("rl_1").state == "closing"       # billing has stopped; the engine is shutting down
    assert client.rl.stop("rl_1", wait=True).state == "closed"


def test_a_live_run_samples_a_batch_and_trains_on_rewards_without_sending_tokens_back(make_client, tmp_path):
    posted = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1/training/sessions/rl_1/tinker/nextron/v1")
        assert request.headers["x-api-key"] == "tml-nk_test"
        if request.method == "POST":
            posted[path] = body(request) if request.content else None
        if path == "/sample":
            return httpx.Response(200, json={"batch": "b1", "prompts": 2})
        if path == "/batches/b1":
            groups = [[{"id": f"b1:{i}:{j}", "text": f"reply {i}{j}", "finish_reason": "stop", "tokens": 3} for j in range(2)] for i in range(2)]
            return httpx.Response(200, json={"done": True, "finished": 2, "prompts": 2, "generate_seconds": 1.5, "samples": groups})
        if path == "/train":
            return httpx.Response(200, json={"step": "s1"})
        if path == "/steps/s1":
            return httpx.Response(200, json={"done": True, "metrics": {
                "step": 1, "episodes": 4, "optimizer_steps": 1, "reward_mean": 0.5, "reward_std": 0.5, "groups": 2,
                "zero_variance_fraction": 0.0, "all_best_fraction": 0.0, "all_worst_fraction": 0.0, "replies": 4,
                "trained_tokens": 12, "kl_sample_train": 0.0, "step_s": 30.0, "generation_s": 12.0, "train_s": 15.0,
                "client_s": 3.0, "session_s": 40.0}})
        return httpx.Response(200, json={})

    with make_client(handler).rl.connect("rl_1", log_dir=tmp_path, log_groups=1) as live:
        batch = live.sample(["2 + 2?", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "3 + 3?"}]], n=2, max_tokens=64)
        assert [[s.text for s in group] for group in batch] == [["reply 00", "reply 01"], ["reply 10", "reply 11"]]
        step = live.train(batch, [[1.0, 0.0], [0.0, 1.0]], learning_rate=2e-6)
        live.finish()
    assert posted["/sample"]["prompts"][0] == {"messages": [{"role": "user", "content": "2 + 2?"}]} and posted["/sample"]["n"] == 2
    assert posted["/train"]["episodes"][:2] == [{"samples": ["b1:0:0"], "reward": 1.0}, {"samples": ["b1:0:1"], "reward": 0.0}]
    assert posted["/train"]["learning_rate"] == 2e-6 and (step.step, step.client_s) == (1, 3.0) and "/finish" in posted
    (logged,) = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]   # the first group only
    assert logged["step"] == 1 and logged["episodes"][0] == {"reward": 1.0, "turns": [
        {"prompt": {"messages": [{"role": "user", "content": "2 + 2?"}]}, "reply": "reply 00", "finish_reason": "stop", "tokens": 3}]}
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["reward_mean"] == 0.5


def test_a_live_run_takes_conversations_as_episodes_and_reports_a_refused_step(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/train"):
            assert body(request)["episodes"] == [{"samples": ["b1:0:0", "b2:0:0"], "reward": 1.0, "group": "task-7"}]
            return httpx.Response(400, json={"error": "ValueError: these replies were sampled before the last training step: sample again"})
        return httpx.Response(200, json={})

    from nextron_cloud import APIError, Episode
    live = make_client(handler).rl.connect("rl_1")
    with pytest.raises(APIError, match="sample again"):
        live.train(episodes=[Episode(samples=["b1:0:0", "b2:0:0"], reward=1.0, group="task-7")], learning_rate=1e-6)
    live.close()


def test_a_live_run_refuses_a_sample_call_that_cannot_work_before_sending_it(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sample"):
            pytest.fail("nothing should be sampled")
        return httpx.Response(200, json={})

    with make_client(handler).rl.connect("rl_1") as live:
        with pytest.raises(TypeError, match=r"give a single prompt as \[prompt\]"):
            live.sample("What is 2+2?")   # would be one prompt per character
        with pytest.raises(TypeError, match=r"not role, content.*give it as \[messages\]"):
            live.sample([{"role": "user", "content": "What is 2+2?"}])   # one conversation, not a list of prompts
        with pytest.raises(ValueError, match="temperature above 0"):
            live.sample(["What is 2+2?"], n=2, temperature=0)


def test_an_rl_runs_metrics_are_its_training_calls(make_client):
    record = {"step": 1, "ts": 1794693600.0, "step_s": 45.0, "generation_s": 16.0, "train_s": 17.0, "client_s": 12.0,
              "session_s": 60.0, "replies": 512, "reward_mean": 0.31, "row_fill": 0.9}
    client = make_client(lambda request: httpx.Response(200, json=[record]))
    (step,) = client.rl.metrics("rl_1")
    assert (step.step, step.client_s, step.reward_mean, step.kl_sample_train) == (1, 12.0, 0.31, None)


def test_an_rl_runs_rollouts_show_a_training_calls_best_and_worst_group(make_client):
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url).removeprefix("https://api.test"))
        group = {"group": "g", "reward_mean": 1.0, "episodes": [{"reward": 1.0, "advantage": 0.5, "turns": [{"prompt": [], "reply": "4", "truncated": False, "tokens": 2}]}]}
        return httpx.Response(200, json={"steps": [1, 2], "step": 2, "groups": [group]})

    shown = make_client(handler).rl.rollouts("rl_1", step=2)
    assert asked == ["/v1/training/sessions/rl_1/rollouts?step=2"] and (shown.steps, shown.groups[0]["episodes"][0]["turns"][0]["reply"]) == ([1, 2], "4")


def test_an_inference_run_serves_a_jobs_model_and_samples_through_its_own_path(make_client, monkeypatch, capsys):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        run = {"id": "inf_1", "state": "ready", "model": "qwen3.8-27b", "params": {"job_id": "job_1", "max_hours": 1},
               "created": 1.0, "updated": 1.0}
        if request.url.path == "/v1/inference/estimate":
            return httpx.Response(200, json={"price_per_gpu_hour": 350, "gpus": 8, "price_per_hour": 2800, "credits": 2800,
                                             "balance": 10000, "enough": True})
        if request.url.path == "/v1/inference/inf_1/nextron/v1/sample":
            return httpx.Response(200, json={"batch": "b1", "prompts": 1})
        if request.url.path == "/v1/inference/inf_1/nextron/v1/batches/b1":
            return httpx.Response(200, json={"done": True, "finished": 1, "prompts": 1, "samples": [[
                {"id": "b1:0:0", "text": "hello", "finish_reason": "stop", "tokens": 2}]]})
        return httpx.Response(200, json=run if request.url.path.startswith("/v1/inference") else {})

    from nextron_cloud.cli import main
    client = make_client(handler)
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    monkeypatch.setenv("NEXTRON_BASE_URL", "https://api.test")
    with pytest.raises(SystemExit) as stopped:   # without --yes: the price only
        main(["inference", "start", "--job", "job_1", "--max-hours", "1"])
    assert stopped.value.code == 3 and json.loads(capsys.readouterr().out)["price"] == "$3.50 per H100-hour (8 GPUs)"
    run = client.inference.start(job_id="job_1", max_hours=1)
    assert (run.id, run.params["job_id"]) == ("inf_1", "job_1") and ("POST", "/v1/inference") in seen
    with client.inference.connect("inf_1") as live:
        assert live.sample(["hi"])[0][0].text == "hello"
        with pytest.raises(NexTronError, match="inf_1 is an inference run"):
            live.train(episodes=[], learning_rate=1e-6)
    assert client.inference.openai_base_url("inf_1") == "https://api.test/v1/inference/inf_1/v1"


def test_a_live_run_keeps_polling_through_a_restart_of_the_api(make_client):
    answers = iter([httpx.Response(500, json={"error": "internal_error"}), httpx.Response(503),
                    httpx.Response(200, json={"done": True, "finished": 1, "prompts": 1, "samples": [[
                        {"id": "b1:0:0", "text": "hi", "finish_reason": "stop", "tokens": 1}]]})])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sample"):
            return httpx.Response(200, json={"batch": "b1", "prompts": 1})
        if request.url.path.endswith("/batches/b1") and "wait" in request.url.query.decode():
            return next(answers)
        return httpx.Response(200, json={"done": True, "finished": 1, "prompts": 1, "samples": [[
            {"id": "b1:0:0", "text": "hi", "finish_reason": "stop", "tokens": 1}]]})

    with make_client(handler).rl.connect("rl_1") as live:
        assert live.sample(["hello"])[0][0].text == "hi"


def test_a_live_poll_is_no_longer_than_the_time_left(make_client):
    holds = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sample"):
            return httpx.Response(200, json={"batch": "b1", "prompts": 1})
        if "wait" in request.url.params:
            holds.append(request.url.params["wait"])
            assert request.extensions["timeout"]["read"] <= 5   # nor may its answer stall for the client's own 60 s
        return httpx.Response(200, json={"done": True, "finished": 1, "prompts": 1, "samples": [[
            {"id": "b1:0:0", "text": "hi", "finish_reason": "stop", "tokens": 1}]]})

    with make_client(handler).rl.connect("rl_1") as live:
        live.sample(["hello"], timeout=5)
    assert holds == ["5"]


def test_a_poll_cut_at_the_deadline_is_a_timeout_like_any_other(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sample"):
            return httpx.Response(200, json={"batch": "b1", "prompts": 1})
        if "/batches/" in request.url.path:
            raise httpx.ReadTimeout("the answer stalled", request=request)
        return httpx.Response(200, json={})

    with make_client(handler).rl.connect("rl_1") as live:
        with pytest.raises(TimeoutError, match="still not done after 0.3 s"):
            live.sample(["hello"], timeout=0.3)


def test_a_timeout_also_ends_the_wait_through_an_outage(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sample"):
            return httpx.Response(200, json={"batch": "b1", "prompts": 1})
        return httpx.Response(503 if "/batches/" in request.url.path else 200, json={})

    with make_client(handler).rl.connect("rl_1") as live:
        started = time.monotonic()
        with pytest.raises(APIError):
            live.sample(["hello"], timeout=0.3)
    assert time.monotonic() - started < 5   # not the two minutes an outage is otherwise waited out


@pytest.mark.parametrize("kind, name", [("image", "cat.png"), ("video", "race.mp4")])
def test_a_live_run_sends_the_images_and_videos_of_its_prompts_before_it_samples(make_client, tmp_path, kind, name):
    import hashlib
    import io
    import struct
    import tarfile

    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 640, 480)
    file = png if kind == "image" else mp4(64, 48, 2)
    (tmp_path / name).write_bytes(file)
    sha = hashlib.sha256(file).hexdigest()
    calls, uploaded, missing = [], {}, [[sha], [sha], []]   # asked for the prompt, then until the run has the file

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(f"{request.method} {path.removeprefix('/v1/training/sessions/rl_1')}")
        if path.endswith("/images/missing"):
            return httpx.Response(200, json={"missing": missing.pop(0)})
        if path == "/v1/datasets":
            assert body(request)["images"] is True
            return httpx.Response(201, json={"id": "ds_1", "upload_url": "https://storage.test/put", "upload_headers": {}})
        if request.url.host == "storage.test":
            uploaded["tar"] = request.content
            return httpx.Response(200)
        if path in ("/v1/datasets/ds_1/complete", "/v1/datasets/ds_1"):
            return httpx.Response(200, json={"id": "ds_1", "name": "images of rl_1", "state": "ready", "bytes": 1, "created": 0, "images": 1,
                                             "kind": "images"})
        if path == "/v1/training/sessions/rl_1/images":
            assert body(request) == {"dataset_id": "ds_1"}
            return httpx.Response(200, json={})
        if path.endswith("/sample"):
            uploaded["prompts"] = body(request)["prompts"]
            return httpx.Response(200, json={"batch": "b1", "prompts": 1})
        if path.endswith("/batches/b1"):
            return httpx.Response(200, json={"done": True, "finished": 1, "prompts": 1, "samples": [[
                {"id": "b1:0:0", "text": "a cat", "finish_reason": "stop", "tokens": 2}]]})
        return httpx.Response(200, json={})

    part = {"type": kind, kind: str(tmp_path / name)}
    messages = [{"role": "user", "content": [part, {"type": "text", "text": "What is this?"}]}]
    with make_client(handler).rl.connect("rl_1") as live:
        assert live.sample([messages])[0][0].text == "a cat"
    with tarfile.open(fileobj=io.BytesIO(uploaded["tar"])) as tar:
        assert tar.getnames() == [f"{kind}s/{sha}.{name.split('.')[1]}"]
    assert uploaded["prompts"][0]["messages"][0]["content"][0] == {"type": kind, kind: f"sha256:{sha}"}
    assert messages[0]["content"][0] == part   # the caller's prompt keeps its path
    wanted = ["POST /tinker/nextron/v1/images/missing", "POST /v1/datasets", "POST /images", "POST /tinker/nextron/v1/sample"]
    assert [c for c in calls if c in wanted] == [wanted[0], wanted[1], wanted[2], wanted[0], wanted[0], wanted[3]]


def test_images_checked_before_a_run_are_attached_without_a_second_upload(make_client, tmp_path):
    import hashlib
    import struct

    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 64, 48)
    (tmp_path / "clock.png").write_bytes(png)
    sha = hashlib.sha256(png).hexdigest()
    calls, missing = [], [[sha], []]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(f"{request.method} {path}")
        if path.endswith("/images/missing"):
            return httpx.Response(200, json={"missing": missing.pop(0)})
        if path == "/v1/datasets":
            return httpx.Response(201, json={"id": "ds_1", "upload_url": "https://storage.test/put", "upload_headers": {}})
        if path in ("/v1/datasets/ds_1/complete", "/v1/datasets/ds_1"):
            return httpx.Response(200, json={"id": "ds_1", "name": "images", "state": "ready", "bytes": 1, "created": 0, "images": 1,
                                             "kind": "images"})
        return httpx.Response(200, json={})

    client = make_client(handler)
    media = client.datasets.upload_images([tmp_path / "clock.png"])
    with client.inference.connect("inf_1") as live:
        live.add_images([tmp_path / "clock.png"], dataset_id=media.id)
    assert calls.count("POST /v1/datasets") == 1 and "POST /v1/inference/inf_1/images" in calls


def test_a_run_billed_by_time_has_a_bill(make_client):
    bill = {"job_id": "rl_1", "state": "succeeded", "billing": "settled", "price_per_hour": 2800, "seconds_charged": 345,
            "held_credits": 700, "charged_credits": 269, "settled_at": 9.0}
    billing = make_client(lambda request: httpx.Response(200, json=bill)).jobs.billing("rl_1")
    assert (billing.price_per_hour, billing.seconds_charged, billing.charged_credits, billing.tokens_charged) == (2800, 345, 269, None)


@pytest.mark.parametrize("command", [["rl", "start", "--model", "qwen3.8-27b"], ["inference", "start", "--job", "job_1"]])
@pytest.mark.parametrize("hours", ["-1", "0", "nan"])
def test_cli_refuses_a_run_length_that_is_not_above_zero_before_asking_its_price(make_client, monkeypatch, capsys, command, hours):
    from nextron_cloud.cli import main
    make_client(lambda request: pytest.fail("nothing should be sent"))
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    with pytest.raises(SystemExit) as stop:
        main(command + ["--max-hours", hours])
    assert stop.value.code == 2 and "--max-hours: must be a number above 0" in capsys.readouterr().err


def test_cli_upload_exits_1_when_the_servers_check_refuses_the_dataset(make_client, monkeypatch, capsys, tmp_path: Path):
    from nextron_cloud.cli import main
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"prompt": "1+1?", "completion": "2"}) + "\n")
    dataset = {"id": "ds_1", "name": "train.jsonl", "bytes": 1, "created": 1.0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/datasets":
            return httpx.Response(201, json={"id": "ds_1", "upload_url": "https://s3.test/1", "upload_headers": {}})
        if request.method == "GET":
            return httpx.Response(200, json=dataset | {"state": "invalid", "errors": ["1 of the images cannot be used"]})
        return httpx.Response(200, json=dataset | {"state": "checking"})

    make_client(handler)
    monkeypatch.setenv("NEXTRON_API_KEY", "nk_test")
    monkeypatch.setenv("NEXTRON_BASE_URL", "https://api.test")
    with pytest.raises(SystemExit) as stop:
        main(["datasets", "upload", str(path)])
    printed = json.loads(capsys.readouterr().out)
    assert stop.value.code == 1 and (printed["id"], printed["errors"]) == ("ds_1", ["1 of the images cannot be used"])


def test_cli_prints_a_missing_file_as_an_error_not_a_traceback(capsys, tmp_path: Path):
    from nextron_cloud.cli import main
    with pytest.raises(SystemExit) as stop:
        main(["datasets", "check", str(tmp_path / "missing.jsonl")])
    assert stop.value.code == 1 and "missing.jsonl" in json.loads(capsys.readouterr().out)["error"]


def test_cli_guide_help_names_every_topic(capsys):
    from nextron_cloud import cli
    with pytest.raises(SystemExit):
        cli.main(["guide", "--help"])
    shown = capsys.readouterr().out
    for topic in sorted(path.stem for path in (Path(__file__).parents[1] / "skills/nextron-cloud/references").glob("*.md")):
        assert topic in shown


def test_max_seq_len_takes_the_longest_context_a_model_trains():
    from typing import get_args, get_type_hints

    from nextron_cloud.types import TrainingParams
    assert 1048576 in get_args(get_type_hints(TrainingParams)["max_seq_len"])
