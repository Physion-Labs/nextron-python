"""`nextron-cloud`: the SDK from a shell. Every command prints JSON."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ._client import APIError, NexTron, NexTronError
from . import experiments
from .data import check_file, prepare_file
from .types import TrainingParams, Wandb


def _print(value: Any) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _value(text: str) -> Any:
    """A value that parses as JSON (numbers, true/false) is taken as that, else as text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _params(pairs: list[str]) -> TrainingParams:
    """`key=value` pairs."""
    params: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param takes key=value, got {pair!r}")
        params[key] = _value(value)
    return params  # type: ignore[return-value]


def _draft(args: argparse.Namespace) -> dict[str, Any]:
    return {"model": args.model, "dataset_id": args.dataset, "layout": args.layout, "params": _params(args.param)}


def _grid(pairs: list[str]) -> dict[str, list[Any]]:
    """`key=v1,v2,...` per axis, each value parsed like --param."""
    axes = {}
    for pair in pairs:
        key, sep, values = pair.partition("=")
        if not sep:
            raise SystemExit(f"--grid takes key=v1,v2,..., got {pair!r}")
        axes[key] = [_value(value) for value in values.split(",")]
    return axes


def _sweep(client: NexTron, args: argparse.Namespace) -> Any:
    rows = experiments.sweep(client, name=args.name, model=args.model, dataset_id=args.dataset, axes=_grid(args.grid),
                             params=_params(args.param), layout=args.layout, start=args.yes)
    if args.yes:
        return {"tag": experiments.sweep_tag(args.name), "jobs": rows}
    unpriced = [row["name"] for row in rows if row["estimate"]["credits"] is None]
    total = None if unpriced else sum(row["estimate"]["credits"] for row in rows)
    _print({"jobs": rows, "total_credits": total, "unpriced": unpriced, "created": False,
            "next": "show the total to the owner; rerun with --yes to start every job"})
    sys.exit(3)


def _report(client: NexTron, args: argparse.Namespace) -> Any:
    if not args.tag and not args.job_id:
        raise SystemExit("report needs --tag or job ids")
    rows = experiments.report(client, tag=args.tag, job_ids=args.job_id)
    if not args.table:
        return rows
    columns = ["id", "name", "state", "steps", "loss", "best_val_loss", "best_val_step", "tokens_trained", "charged_credits",
               "training_minutes", "checkpoint_ready"]
    print("\t".join(columns))
    for row in rows:
        print("\t".join("" if row[c] is None else str(row[c]) for c in columns))
    return None


def _datasets_list(client: NexTron, args: argparse.Namespace) -> Any:
    datasets = [d for d in client.datasets.list() if d.sample or not args.sample]
    if not args.table:
        return datasets
    max_seq_len = None
    if args.model:
        models = {m.id: m for m in client.models()}
        if args.model not in models:
            raise SystemExit(f"unknown model {args.model}; see nextron-cloud models")
        max_seq_len = max(method.max_seq_len for method in models[args.model].methods.values())
    over_column = f"over_{max_seq_len}" if max_seq_len else "over_max_seq_len"
    print("\t".join(["id", "name", "state", "samples", "validation", "tokens", "longest", over_column]))
    for d in datasets:
        stats = (d.tokens or {}).get(args.model or "", {})
        tokens = stats.get("total", d.est_tokens)
        over = stats.get("over", {}).get(str(max_seq_len))
        row = [d.id, d.name, d.state, d.samples, d.validation, tokens, stats.get("max"), over]
        print("\t".join("" if v is None else str(v) for v in row))
    return None


def _serve_rewards(client: NexTron, args: argparse.Namespace) -> Any:
    import importlib.util
    path, _, name = args.reward.rpartition(":")
    if not path or not name:
        raise SystemExit("give the reward as FILE.py:FUNCTION")
    spec = importlib.util.spec_from_file_location("reward_module", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"batches_scored": client.jobs.serve_rewards(args.job_id, getattr(module, name))}


def _create(client: NexTron, args: argparse.Namespace) -> Any:
    draft = _draft(args)
    if not args.yes:
        estimate = client.jobs.estimate(**draft)
        _print({"estimate": estimate.model_dump(mode="json"), "created": False,
                "next": "show the cost to the owner; rerun with --yes to start the job"})
        sys.exit(3)
    wandb: Wandb | None = None
    if args.wandb_project:
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("--wandb-project needs WANDB_API_KEY in the environment")
        wandb = {"api_key": os.environ["WANDB_API_KEY"], "project": args.wandb_project}
        if args.wandb_run_name:
            wandb["run_name"] = args.wandb_run_name
    return client.jobs.create(**draft, name=args.name, tags=args.tag or None, wandb=wandb, idempotency_key=args.idempotency_key)


def _rl_start(client: NexTron, args: argparse.Namespace) -> Any:
    if not args.yes:
        estimate = client.rl.estimate(model=args.model, max_hours=args.max_hours)
        if estimate.price_per_gpu_hour is None:   # no price: the model has no RL runs
            raise SystemExit(f"{args.model} has no Serverless RL; `nextron-cloud models` lists each model's methods")
        price = f"${estimate.price_per_gpu_hour / 100:.2f} per H100-hour ({estimate.gpus} GPUs)"
        _print({"price": price, "estimate": estimate.model_dump(mode="json"), "created": False,
                "next": "show the cost to the owner; rerun with --yes to start the RL run"})
        sys.exit(3)
    return client.rl.start(model=args.model, max_hours=args.max_hours, max_context=args.max_context,
                           max_grad_norm=args.max_grad_norm, grad_clip=False if args.no_grad_clip else None,
                           idle_timeout=args.idle_timeout, name=args.name, tags=args.tag)


def _rl_wait(client: NexTron, args: argparse.Namespace) -> Any:
    """Progress goes to stderr about once a minute; the ready run to stdout."""
    started, shown = time.monotonic(), [0.0]

    def progress(run: Any) -> None:
        elapsed = time.monotonic() - started
        if elapsed - shown[0] >= 60:
            shown[0] = elapsed
            print(f"{run.state}, {elapsed:.0f} s (loading the model takes 6-8 minutes, about 20 for a two-node model)", file=sys.stderr, flush=True)
    return client.rl.wait(args.run_id, timeout=args.timeout, on_poll=progress)


def _rl_env(client: NexTron, args: argparse.Namespace) -> None:
    """Shell lines for `eval`. The key is not printed: the line refers to NEXTRON_API_KEY."""
    url = client.rl.env(args.run_id)["TINKER_BASE_URL"]
    print(f"export TINKER_BASE_URL={url}")
    print('export TINKER_API_KEY="tml-$NEXTRON_API_KEY"')


def _inference_start(client: NexTron, args: argparse.Namespace) -> Any:
    if not args.yes:
        estimate = client.inference.estimate(job_id=args.job, max_hours=args.max_hours)
        if estimate.price_per_gpu_hour is None:   # no price: the job's model has no inference runs
            raise SystemExit(f"the model of {args.job} has no inference; `nextron-cloud models` lists each model's methods")
        price = f"${estimate.price_per_gpu_hour / 100:.2f} per H100-hour ({estimate.gpus} GPUs)"
        _print({"price": price, "estimate": estimate.model_dump(mode="json"), "created": False,
                "next": "show the cost to the owner; rerun with --yes to start the inference run"})
        sys.exit(3)
    return client.inference.start(job_id=args.job, max_hours=args.max_hours, idle_timeout=args.idle_timeout,
                                  name=args.name, tags=args.tag)


def _inference_wait(client: NexTron, args: argparse.Namespace) -> Any:
    """Progress goes to stderr about once a minute; the ready run to stdout."""
    started, shown = time.monotonic(), [0.0]

    def progress(run: Any) -> None:
        elapsed = time.monotonic() - started
        if elapsed - shown[0] >= 60:
            shown[0] = elapsed
            print(f"{run.state}, {elapsed:.0f} s (loading the model takes about 8 minutes)", file=sys.stderr, flush=True)
    return client.inference.wait(args.run_id, timeout=args.timeout, on_poll=progress)


def _guide_dir() -> Path:
    """The guide inside the installed package, or in the repository for a development install."""
    packaged = Path(__file__).parent / "guide"
    return packaged if packaged.exists() else Path(__file__).parents[2] / "skills" / "nextron-cloud"


def _guide(args: argparse.Namespace) -> tuple[None, int]:
    root = _guide_dir()
    path = root / "SKILL.md" if not args.topic else root / "references" / f"{args.topic}.md"
    if not path.exists():
        topics = sorted(p.stem for p in (root / "references").glob("*.md"))
        raise SystemExit(f"no guide topic {args.topic!r}; topics: {', '.join(topics)}")
    print(path.read_text(encoding="utf-8"))
    return None, 0


def _events(client: NexTron, args: argparse.Namespace) -> None:
    for event in client.jobs.events(args.job_id, after=args.after, follow=args.follow, until=args.until):
        print(json.dumps(event.model_dump(mode="json"), ensure_ascii=False), flush=True)


def _logs(client: NexTron, args: argparse.Namespace) -> None:
    for line in client.jobs.logs(args.job_id, follow=args.follow):
        print(line.line, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nextron-cloud", description="NexTron Cloud from the shell. Reads NEXTRON_API_KEY.")
    groups = parser.add_subparsers(dest="group", required=True)

    cmd = groups.add_parser("guide", help="how to run experiments: read this first (no key needed)")
    cmd.add_argument("topic", nargs="?", help="data, training, monitoring, checkpoints, rl, serverless-rl or api")
    cmd.set_defaults(run=None, local=_guide)
    groups.add_parser("me", help="account, balance (US cents), price list").set_defaults(run=lambda c, a: c.me())
    groups.add_parser("models", help="models a job can train, live").set_defaults(run=lambda c, a: c.models())
    groups.add_parser("layouts", help="GPU setups").set_defaults(run=lambda c, a: c.layouts())

    datasets = groups.add_parser("datasets", help="upload, import, check and list training data").add_subparsers(dest="cmd", required=True)
    cmd = datasets.add_parser("list", help="your datasets and the sample datasets every account can use")
    cmd.add_argument("--sample", action="store_true", help="only the sample datasets")
    cmd.add_argument("--table", action="store_true", help="tab-separated: samples, tokens, longest sample")
    cmd.add_argument("--model", help="with --table: tokens and longest sample for this model, and how many samples are "
                     "over its max_seq_len (they would be truncated)")
    cmd.set_defaults(run=_datasets_list)
    cmd = datasets.add_parser("get")
    cmd.add_argument("dataset_id")
    cmd.set_defaults(run=lambda c, a: c.datasets.get(a.dataset_id))
    cmd = datasets.add_parser("check", help="the server's format check, locally; no key needed")
    cmd.add_argument("path")
    cmd.set_defaults(run=None, local=lambda a: _checked(check_file(a.path)))
    cmd = datasets.add_parser("prepare", help="convert a .jsonl/.json/.csv file to train.jsonl and val.jsonl; no key needed")
    cmd.add_argument("path")
    cmd.add_argument("--out-dir", default=".")
    cmd.add_argument("--val-fraction", type=float, default=0.1)
    cmd.add_argument("--seed", type=int, default=0)
    cmd.add_argument("--prompt-column", action="append", help="the user message's column (with --response-column); "
                     "repeat to join several, e.g. instruction and context")
    cmd.add_argument("--response-column", help="the assistant message's column")
    cmd.add_argument("--answer-column", help="instead of --response-column: the answer's column, for RL (grpo) samples")
    cmd.add_argument("--system", help="a system message for every sample")
    cmd.set_defaults(run=None, local=lambda a: (prepare_file(
        a.path, a.out_dir, val_fraction=a.val_fraction, seed=a.seed, prompt_column=a.prompt_column,
        response_column=a.response_column, answer_column=a.answer_column, system=a.system), 0))
    cmd = datasets.add_parser("upload", help="check, upload and wait for the server's check")
    cmd.add_argument("path")
    cmd.add_argument("--name")
    cmd.set_defaults(run=lambda c, a: c.datasets.wait(c.datasets.upload(a.path, name=a.name).id))
    cmd = datasets.add_parser("splits", help="a Hugging Face dataset's configs and splits")
    cmd.add_argument("url")
    cmd.set_defaults(run=lambda c, a: c.datasets.huggingface_splits(a.url))
    cmd = datasets.add_parser("import", help="import a Hugging Face split and wait for its check")
    cmd.add_argument("url")
    cmd.add_argument("--config", default="default")
    cmd.add_argument("--split", default="train")
    cmd.add_argument("--name")
    cmd.set_defaults(run=lambda c, a: c.datasets.wait(
        c.datasets.import_huggingface(a.url, config=a.config, split=a.split, name=a.name).id))
    cmd = datasets.add_parser("download-url")
    cmd.add_argument("dataset_id")
    cmd.set_defaults(run=lambda c, a: {"url": c.datasets.download_url(a.dataset_id)})
    cmd = datasets.add_parser("delete")
    cmd.add_argument("dataset_id")
    cmd.set_defaults(run=lambda c, a: c.datasets.delete(a.dataset_id))

    jobs = groups.add_parser("jobs", help="estimate, start and watch training jobs").add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "create"):
        cmd = jobs.add_parser(name, help="cost of a job" if name == "estimate" else "estimate, then start with --yes")
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--dataset", required=True, help="dataset id")
        cmd.add_argument("--layout", default="auto")
        cmd.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="a training param; repeat")
        if name == "estimate":
            cmd.set_defaults(run=lambda c, a: c.jobs.estimate(**_draft(a)))
            continue
        cmd.add_argument("--yes", action="store_true", help="start the job; without it only the estimate is printed")
        cmd.add_argument("--name", help="your label for the job")
        cmd.add_argument("--tag", action="append", help="a tag to group jobs by; repeat")
        cmd.add_argument("--idempotency-key", help="the same key never starts a second job")
        cmd.add_argument("--wandb-project", help="log to this W&B project; the key comes from WANDB_API_KEY")
        cmd.add_argument("--wandb-run-name")
        cmd.set_defaults(run=_create)
    cmd = jobs.add_parser("list")
    cmd.add_argument("--tag")
    cmd.set_defaults(run=lambda c, a: c.jobs.list(tag=a.tag))
    for name, run in (("get", lambda c, a: c.jobs.get(a.job_id)), ("wait", lambda c, a: c.jobs.wait(a.job_id)),
                      ("wait-checkpoint", lambda c, a: c.jobs.wait_checkpoint(a.job_id)),
                      ("progress", lambda c, a: c.jobs.progress(a.job_id)), ("billing", lambda c, a: c.jobs.billing(a.job_id)),
                      ("metrics", lambda c, a: c.jobs.metrics(a.job_id)), ("cancel", lambda c, a: c.jobs.cancel(a.job_id)),
                      ("checkpoint", lambda c, a: c.jobs.checkpoint(a.job_id))):
        cmd = jobs.add_parser(name)
        cmd.add_argument("job_id")
        cmd.set_defaults(run=run)
    cmd = jobs.add_parser("events", help="the job's events as JSON lines")
    cmd.add_argument("job_id")
    cmd.add_argument("--after", type=int, default=0, help="the last seq already read")
    cmd.add_argument("--follow", action="store_true", help="keep reading until training is over")
    cmd.add_argument("--until", choices=["training", "checkpoint"], default="training")
    cmd.set_defaults(run=_events)
    cmd = jobs.add_parser("serve-rewards", help="score a remote-reward GRPO job's completions with your function until it ends")
    cmd.add_argument("job_id")
    cmd.add_argument("reward", metavar="FILE.py:FUNCTION", help="reward(item) -> float; item has messages, completion, row, "
                     "answer, metadata")
    cmd.set_defaults(run=_serve_rewards)
    cmd = jobs.add_parser("logs")
    cmd.add_argument("job_id")
    cmd.add_argument("--follow", action="store_true", help="keep reading until the job ends")
    cmd.set_defaults(run=_logs)
    cmd = jobs.add_parser("download", help="download the checkpoint and check every file's sha256")
    cmd.add_argument("job_id")
    cmd.add_argument("dest")
    cmd.add_argument("--include", action="append", metavar="GLOB", help="only files matching this pattern; repeat")
    cmd.set_defaults(run=lambda c, a: {"path": str(c.jobs.download_checkpoint(a.job_id, a.dest, include=a.include))})
    cmd = jobs.add_parser("export", help="save the job record as a zip")
    cmd.add_argument("job_id")
    cmd.add_argument("dest")
    cmd.set_defaults(run=lambda c, a: {"path": str(c.jobs.export(a.job_id, a.dest))})

    rl = groups.add_parser("rl", help="Serverless RL: your own loop samples and trains on NexTron's GPUs").add_subparsers(dest="cmd", required=True)
    cmd = rl.add_parser("start", help="price for --max-hours, then start with --yes")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--max-hours", type=float, required=True, help="the run stops itself after this long")
    cmd.add_argument("--max-context", type=int, help="tokens: a prompt with its reply, and a training sequence (default 16384)")
    cmd.add_argument("--max-grad-norm", type=float, help="gradient clipping for every step (default 1.0)")
    cmd.add_argument("--no-grad-clip", action="store_true", help="no gradient clipping: a step is about 2 s shorter")
    cmd.add_argument("--idle-timeout", type=int, help="seconds without a sample or a step before it stops (default 1800)")
    cmd.add_argument("--name", help="your label for the run")
    cmd.add_argument("--tag", action="append", help="a tag; repeat")
    cmd.add_argument("--yes", action="store_true", help="start the run; without it only the price is printed")
    cmd.set_defaults(run=_rl_start)
    rl.add_parser("list").set_defaults(run=lambda c, a: c.rl.list())
    cmd = rl.add_parser("metrics", help="the record of every training call: rewards, time split, row fill")
    cmd.add_argument("run_id")
    cmd.set_defaults(run=lambda c, a: c.rl.metrics(a.run_id))
    cmd = rl.add_parser("rollouts", help="what the model wrote in a training call: its best and worst group, with rewards")
    cmd.add_argument("run_id")
    cmd.add_argument("--step", type=int, help="the training call; the latest without")
    cmd.set_defaults(run=lambda c, a: c.rl.rollouts(a.run_id, step=a.step))
    cmd = rl.add_parser("wait", help="until the model is loaded (6-8 minutes; about 20 for qwen3.8-next-180b): progress on stderr, then the run")
    cmd.add_argument("run_id")
    cmd.add_argument("--timeout", type=float, help="give up after this many seconds (the run goes on)")
    cmd.set_defaults(run=_rl_wait)
    cmd = rl.add_parser("stop", help="stop billing; the model is saved if it trained")
    cmd.add_argument("run_id")
    cmd.add_argument("--wait", action="store_true", help="return once it is closed and the GPUs are released (about a minute)")
    cmd.set_defaults(run=lambda c, a: c.rl.stop(a.run_id, wait=a.wait))
    for name, run, text in (("get", lambda c, a: c.rl.get(a.run_id), None),
                            ("env", _rl_env, 'TINKER_BASE_URL and TINKER_API_KEY for the tinker SDK: eval "$(... env ID)"')):
        cmd = rl.add_parser(name, help=text)
        cmd.add_argument("run_id")
        cmd.set_defaults(run=run)

    inference = groups.add_parser("inference", help="serve a trained job's model on NexTron's GPUs and call it").add_subparsers(dest="cmd", required=True)
    cmd = inference.add_parser("start", help="price for --max-hours, then start with --yes")
    cmd.add_argument("--job", required=True, help="a training job that succeeded in the last 7 days")
    cmd.add_argument("--max-hours", type=float, required=True, help="the run stops itself after this long")
    cmd.add_argument("--idle-timeout", type=int, help="seconds without a request before it stops (default 1800)")
    cmd.add_argument("--name", help="your label for the run")
    cmd.add_argument("--tag", action="append", help="a tag; repeat")
    cmd.add_argument("--yes", action="store_true", help="start the run; without it only the price is printed")
    cmd.set_defaults(run=_inference_start)
    inference.add_parser("list").set_defaults(run=lambda c, a: c.inference.list())
    cmd = inference.add_parser("wait", help="until the model is loaded, about 8 minutes: progress on stderr, then the run")
    cmd.add_argument("run_id")
    cmd.add_argument("--timeout", type=float, help="give up after this many seconds (the run goes on)")
    cmd.set_defaults(run=_inference_wait)
    cmd = inference.add_parser("stop", help="stop billing; the GPUs are released about a minute later")
    cmd.add_argument("run_id")
    cmd.add_argument("--wait", action="store_true", help="return once it is closed")
    cmd.set_defaults(run=lambda c, a: c.inference.stop(a.run_id, wait=a.wait))
    for name, run, text in (("get", lambda c, a: c.inference.get(a.run_id), None),
                            ("url", lambda c, a: {"openai_base_url": c.inference.openai_base_url(a.run_id)},
                             "the base URL for an OpenAI client; its api_key is your NexTron key")):
        cmd = inference.add_parser(name, help=text)
        cmd.add_argument("run_id")
        cmd.set_defaults(run=run)

    cmd = groups.add_parser("sweep", help="one job per combination of --grid values; estimates, then starts with --yes")
    cmd.add_argument("--name", required=True, help="the sweep's name; its jobs are tagged sweep:<name>")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--dataset", required=True, help="dataset id")
    cmd.add_argument("--layout", default="auto")
    cmd.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="a param every job gets; repeat")
    cmd.add_argument("--grid", action="append", required=True, metavar="KEY=V1,V2", help="values to sweep; repeat")
    cmd.add_argument("--yes", action="store_true", help="start the jobs; without it only the estimates are printed")
    cmd.set_defaults(run=_sweep)
    cmd = groups.add_parser("report", help="compare jobs: settings, losses, tokens, time")
    cmd.add_argument("job_id", nargs="*")
    cmd.add_argument("--tag", help="every job with this tag, e.g. sweep:<name>")
    cmd.add_argument("--table", action="store_true", help="tab-separated text instead of JSON")
    cmd.set_defaults(run=_report)
    return parser


def _checked(result: Any) -> tuple[dict[str, Any], int]:
    return {"ok": result.ok, "kind": result.kind, "samples": result.samples, "chars": result.chars, "errors": result.errors}, 0 if result.ok else 1


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.run is None:
        output, code = args.local(args)
        if output is not None:
            _print(output)
        sys.exit(code)
    try:
        with NexTron() as client:
            result = args.run(client, args)
    except APIError as error:
        _print({"error": error.code, "status": error.status_code, "message": error.message})
        sys.exit(1)
    except NexTronError as error:
        _print({"error": str(error)})
        sys.exit(1)
    if result is not None:
        _print(result)


if __name__ == "__main__":
    main()
