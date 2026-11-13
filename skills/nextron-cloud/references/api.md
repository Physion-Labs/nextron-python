# SDK and CLI reference

```python
from nextron_cloud import NexTron, APIError, check_file, to_messages, write_jsonl
client = NexTron(api_key=None, base_url=None, org=None, timeout=60.0, max_retries=3)
```

`api_key` defaults to `NEXTRON_API_KEY`, `base_url` to `NEXTRON_BASE_URL` or https://nextron.physionlabs.ai. `org`
picks an organisation when a login belongs to several (keys belong to one). Use `with NexTron() as client:` to close
connections. Reads, job creation and cancel are retried on connection errors and 429/502/503/504; other errors raise
`APIError` (`status_code`, `code`, `message`, `body`). Responses are pydantic models; unknown fields are kept.

## Account

| call | returns |
|---|---|
| `client.me()` | `org` (`id`, `name`, `role`, `credits`: the balance in US cents), `orgs` (every organisation of the login; no balance), `billing` (`prices`: `{model: {method: cents_per_million_tokens}}`, possibly including retired models; `checkout`, `min_cents`, `max_cents`: website top-ups) |
| `client.models()` | `list[ModelInfo]`, live: `id`, `name`, `methods` (`max_seq_len`, `layouts`, `price_per_mtok`), `layouts` (`gpus`, `tokens_per_s_per_gpu`, `available`, `queued`), `batch_size`, `validation_set` |
| `client.layouts()` | every GPU setup: `id`, `name`, `nodes`, `gpus`, `models`, `tokens_per_s` (per GPU, per model), `available`, `queued`, `coming_soon`. Prefer `models()`, which gives the same per model and method |

## Datasets

| call | does |
|---|---|
| `datasets.upload(path, name=None, check=True)` | local format check, upload, server check queued; returns `Dataset` in `checking` |
| `datasets.import_huggingface(url, config="default", split="train", name=None, hf_token=None)` | import one split; `importing` then `checking` |
| `datasets.huggingface_splits(url, hf_token=None)` | `[{"config", "split", "rows", "bytes"}]` |
| `datasets.wait(id, timeout=3600, poll_interval=5)` | until `ready` or `invalid` |
| `datasets.get(id)`, `datasets.list()` | `Dataset`: `id`, `name`, `state`, `samples`, `est_tokens`, `too_long`, `tokens`, `errors`, `sample`, `validation`, `source` (see data.md) |
| `datasets.download_url(id)` | a one-hour link to the file as training reads it |
| `datasets.delete(id)` | refused (409) while an unfinished job uses it |

## Jobs

| call | does |
|---|---|
| `jobs.estimate(model, dataset_id, layout="auto", params=None)` | `Estimate`: `tokens`, `price_per_mtok`, `credits` (held, US cents), `balance`, `enough`, `metered`, `params`, `steps`, `truncated_samples`, `warnings`; raises `APIError` 422 when the job would be refused |
| `jobs.progress(id)` | `Progress`: `phase`, `jobs_ahead`, `training`, `checkpoint`, `timeline`, `last_activity`, `idle_seconds` (monitoring.md) |
| `jobs.events(id, after=0, follow=False, until="training")` | iterator of `Event` (`seq`, `ts`, `type`, `data`) |
| `jobs.billing(id)` | `Billing`: `billing` (`held`/`settled`/`released`/`not_metered`), `price_per_mtok`, `held_credits`, `charged_credits`, `tokens_charged`, `settled_at` |
| `jobs.create(model, dataset_id, layout="auto", params=None, name=None, tags=None, wandb=None, idempotency_key=None)` | `Job` in `queued`; same key → same job. Tags: up to 20, each up to 64 of `A-Z a-z 0-9 _ . : = / + -` |
| `jobs.get(id)`, `jobs.list(tag=None)` (the whole organisation's, newest first) | `Job`: `id`, `name`, `tags`, `state`, `model`, `layout`, `dataset_id`, `params`, `metrics`, `checkpoint`, `error`, `done`, `checkpoint_ready`; `get` adds `timeline`, `notices`, `tokens_trained`, `config` |
| `jobs.wait(id, timeout=None, poll_interval=30)` | until training is over (`succeeded`, `failed`, `cancelled`); a failed job is returned, not raised |
| `jobs.wait_checkpoint(id, timeout=None, poll_interval=60)` | until a succeeded job's checkpoint is uploaded; `NexTronError` if the job ended without one |
| `jobs.logs(id, follow=False, poll_interval=10)` | iterator of `LogLine` (`seq`, `ts`, `line`) |
| `jobs.metrics(id)` | `list[MetricPoint]`: `step`, `total_steps`, `loss`, `val_loss`, `lr`, `step_time_s`, `tokens_per_s_per_gpu`, `tokens_per_s_total`, `ts`, and `samples`, `real_tokens`, `trained_tokens`, `padding_tokens` (see monitoring.md) |
| `jobs.cancel(id)` | queued: cancelled now; running: stops soon, no checkpoint |
| `jobs.checkpoint(id)` | `Checkpoint`: `step`, `bytes`, `expires`, `files` (`name`, `bytes`, `sha256`, `url`) |
| `jobs.download_checkpoint(id, dest, include=None)` | downloads into `dest` (only files matching the `include` globs), resumes partial files, checks each sha256; 409 while uploading |
| `jobs.export(id, dest)` | the job record as a zip |
| `jobs.manifest(id)`, `jobs.trace(id)`, `jobs.telemetry(id)` | sha256 manifest (bytes), per-rank Chrome trace (bytes), GPU/host utilisation (dict) |

## Data helpers

| call | does |
|---|---|
| `to_messages(sample)` | one sample in any accepted format → `[{"role", "content"}]`; `ValueError` says why not |
| `write_jsonl(samples, path)` | write samples as `messages` JSONL; returns the count; raises on the first bad one |
| `check_file(path)` | `CheckResult(samples, chars, errors)`, `.ok`: the server's format check, locally |
| `data.read_rows(path)` | rows of a `.jsonl`, `.json` or `.csv` file |
| `data.prepare_file(path, out_dir=".", val_fraction=0.1, seed=0, prompt_column=None, response_column=None, system=None)` (`prompt_column`: a name or a list joined by blank lines) | writes `train.jsonl` / `val.jsonl`; returns their paths and counts, `skipped`, `errors` |

## Experiments

| call | does |
|---|---|
| `experiments.sweep(client, name, model, dataset_id, axes, params=None, layout="auto", start=False)` | one job per combination of `axes` (`{"learning_rate": [...], ...}`): estimates, or with `start` creates them named `<name> key=value` and tagged `sweep:<name>`; repeat-safe |
| `experiments.report(client, tag=None, job_ids=None)` | per job: settings, `steps`, `loss`, `val_loss`, `best_val_loss`, `best_val_step`, `tokens_trained`, `charged_credits`, `training_minutes`, `checkpoint_ready`, `error` |
| `experiments.grid(axes)`, `experiments.sweep_tag(name)` | the combinations; the sweep's tag |

## CLI

`nextron-cloud` prints JSON and exits 0 on success, 1 on an error (`{"error", "status", "message"}`), 3 when
`jobs create` or `sweep` printed estimates instead of starting.

```
nextron-cloud me | models | layouts
nextron-cloud datasets list [--sample] [--table [--model M]] | get ID | check PATH | upload PATH [--name N] | splits URL
nextron-cloud datasets prepare PATH [--out-dir D] [--val-fraction F] [--seed S] [--prompt-column C ... --response-column C] [--system TEXT]
nextron-cloud datasets import URL [--config C] [--split S] [--name N] | download-url ID | delete ID
nextron-cloud jobs estimate --model M --dataset ID [--layout L] [--param KEY=VALUE ...]
nextron-cloud jobs create   --model M --dataset ID [--layout L] [--param KEY=VALUE ...] [--yes]
                            [--name N] [--tag T ...] [--idempotency-key K]
                            [--wandb-project P [--wandb-run-name R]]   (W&B key from WANDB_API_KEY)
nextron-cloud jobs list [--tag T] | get ID | wait ID | progress ID | billing ID | metrics ID | cancel ID | logs ID [--follow]
nextron-cloud jobs events ID [--after SEQ] [--follow] [--until training|checkpoint]
nextron-cloud jobs wait-checkpoint ID | checkpoint ID | download ID DEST [--include GLOB ...] | export ID DEST
nextron-cloud sweep --name N --model M --dataset ID [--layout L] [--param KEY=VALUE ...] --grid KEY=V1,V2 [...] [--yes]
nextron-cloud report [JOB_ID ...] [--tag T] [--table]
```

`--param` values are parsed as JSON when they can be (`max_steps=3`, `learning_rate=1e-4`), else taken as text
(`train_type=lora`). `datasets upload` and `datasets import` wait for the server's check.
