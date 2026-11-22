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
| `client.models()` | `list[ModelInfo]`, live: `id`, `name`, `methods` (`max_seq_len`, `layouts`, `price_per_mtok`, `defaults`, `max_tokens_per_gpu`), `layouts` (`gpus`, `tokens_per_s_per_gpu`, `available`, `queued`), `batch_size`, `validation_set` |
| `client.layouts()` | every GPU setup: `id`, `name`, `nodes`, `gpus`, `models`, `tokens_per_s` (per GPU, per model), `available`, `queued`, `coming_soon`. Prefer `models()`, which gives the same per model and method |

## Datasets

| call | does |
|---|---|
| `datasets.upload(path, name=None, check=True)` | local format check, upload, server check queued; returns `Dataset` in `checking` |
| `datasets.import_huggingface(url, config="default", split="train", name=None, hf_token=None)` | import one split; `importing` then `checking` |
| `datasets.huggingface_splits(url, hf_token=None)` | `[{"config", "split", "rows", "bytes"}]` |
| `datasets.wait(id, timeout=3600, poll_interval=5)` | until `ready` or `invalid` |
| `datasets.get(id)`, `datasets.list()` | `Dataset`: `id`, `name`, `state`, `samples`, `est_tokens`, `too_long`, `tokens`, `errors`, `sample`, `validation`, `source`, `kind` (`chat`, `rl` for prompts with answers, or `preference` for DPO pairs) (see data.md) |
| `datasets.download_url(id)` | a one-hour link to the file as training reads it |
| `datasets.delete(id)` | refused (409) while an unfinished job uses it |

## Jobs

| call | does |
|---|---|
| `jobs.estimate(model, dataset_id, layout="auto", params=None)` | `Estimate`: `tokens`, `price_per_mtok`, `credits` (held, US cents), `balance`, `enough`, `metered`, `params`, `steps`, `truncated_samples`, `warnings`, `video` (the video settings, for a dataset with videos); raises `APIError` 422 when the job would be refused |
| `jobs.progress(id)` | `Progress`: `phase`, `jobs_ahead`, `training`, `checkpoint`, `timeline`, `last_activity`, `idle_seconds` (monitoring.md) |
| `jobs.events(id, after=0, follow=False, until="training")` | iterator of `Event` (`seq`, `ts`, `type`, `data`) |
| `jobs.billing(id)` | `Billing`: `billing` (`held`/`settled`/`released`/`not_metered`), `held_credits`, `charged_credits`, `settled_at`; a training job has `price_per_mtok` and `tokens_charged`, an RL or inference run (billed by time; pass its `rl_...` / `inf_...` id) `price_per_hour` and `seconds_charged` |
| `jobs.create(model, dataset_id, layout="auto", params=None, name=None, tags=None, wandb=None, idempotency_key=None)` | `Job` in `queued`; same key → same job. Tags: up to 20, each up to 64 of `A-Z a-z 0-9 _ . : = / + -` |
| `jobs.get(id)`, `jobs.list(tag=None)` (the whole organisation's, newest first) | `Job`: `id`, `name`, `tags`, `state`, `model`, `layout`, `dataset_id`, `params`, `metrics`, `checkpoint`, `error`, `done`, `checkpoint_ready`; `get` adds `timeline`, `notices`, `tokens_trained`, `config` |
| `jobs.wait(id, timeout=None, poll_interval=30)` | until training is over (`succeeded`, `failed`, `cancelled`); a failed job is returned, not raised |
| `jobs.wait_checkpoint(id, timeout=None, poll_interval=60)` | until a succeeded job's checkpoint is uploaded; `NexTronError` if the job ended without one (also an RL run that trained no step) |
| `jobs.logs(id, follow=False, poll_interval=10)` | iterator of `LogLine` (`seq`, `ts`, `line`) |
| `jobs.rollouts(id)`, `jobs.score(id, batch_id, scores)` | remote-reward GRPO: `RolloutBatch` (`id`, `prompts`: row → prompt messages, `items`: completion, row, answer, metadata) waiting for scores; send one score per item (rl.md) |
| `jobs.serve_rewards(id, reward)` | score every batch with `reward(item) -> float` until training is over; returns the batches scored |
| `jobs.metrics(id)` | `list[MetricPoint]`: `step`, `total_steps`, `loss`, `val_loss`, `lr`, `step_time_s`, `tokens_per_s_per_gpu`, `tokens_per_s_total`, `ts`, and `samples`, `real_tokens`, `trained_tokens`, `padding_tokens` (see monitoring.md); GRPO adds `reward_mean`, `zero_variance_fraction`, `completion_length_mean`, `generation_s`, `round_time_s` (rl.md) |
| `jobs.cancel(id)` | queued: cancelled now; running: stops soon, no checkpoint |
| `jobs.checkpoint(id)` | `Checkpoint`: `step`, `bytes`, `expires`, `files` (`name`, `bytes`, `sha256`, `url`) |
| `jobs.download_checkpoint(id, dest, include=None)` | downloads into `dest` (only files matching the `include` globs), resumes partial files, checks each sha256; 409 while uploading |
| `jobs.export(id, dest)` | the job record as a zip |
| `jobs.manifest(id)`, `jobs.trace(id)`, `jobs.telemetry(id)` | sha256 manifest (bytes), per-rank Chrome trace (bytes), GPU/host utilisation (dict) |

## Data helpers

| call | does |
|---|---|
| `to_messages(sample)` | one sample in any accepted format → `[{"role", "content"}]`; `ValueError` says why not |
| `write_jsonl(samples, path)` | write samples as `messages` JSONL (a sample with images or videos as it is); returns the count; raises on the first bad one |
| `check_file(path)` | `CheckResult(samples, chars, errors, kind, images, videos)`, `.ok`: the server's format check, locally |
| `data.read_rows(path)` | rows of a `.jsonl`, `.json` or `.csv` file |
| `data.prepare_file(path, out_dir=".", val_fraction=0.1, seed=0, prompt_column=None, response_column=None, answer_column=None, system=None)` (`prompt_column`: a name or a list joined by blank lines, given with `response_column` or, for RL samples, `answer_column`) | writes `train.jsonl` / `val.jsonl`; returns their paths and counts, `skipped`, `errors` |

## Experiments

| call | does |
|---|---|
| `experiments.sweep(client, name, model, dataset_id, axes, params=None, layout="auto", start=False)` | one job per combination of `axes` (`{"learning_rate": [...], ...}`): estimates, or with `start` creates them named `<name> key=value` and tagged `sweep:<name>`; repeat-safe |
| `experiments.report(client, tag=None, job_ids=None)` | per job: settings, `steps`, `loss`, `val_loss`, `best_val_loss`, `best_val_step`, `tokens_trained`, `charged_credits`, `training_minutes`, `checkpoint_ready`, `error` |
| `experiments.grid(axes)`, `experiments.sweep_tag(name)` | the combinations; the sweep's tag |

## Serverless RL

| call | does |
|---|---|
| `rl.estimate(model, max_hours)` | `RLEstimate`: `price_per_gpu_hour`, `gpus`, `price_per_hour` (the two multiplied), `credits` (held for `max_hours`), `balance`, `enough`; US cents |
| `rl.start(model, max_hours, max_context=None, max_grad_norm=None, grad_clip=None, idle_timeout=None, images=False, video_fps=None, video_max_frames=None, video_frame_tokens=None, name=None, tags=None)` | `RLRun` in `queued`; 422 for a model without Serverless RL. Video settings: media.md |
| `rl.connect(id, log_dir=None, log_groups=4)` | `LiveRun` on a ready run (context manager; keeps the run alive while open). `log_dir`: `metrics.jsonl` and `rollouts.jsonl` (the first `log_groups` groups of every step) are written there |
| `live.sample(prompts, n=1, max_tokens=None, temperature=1.0, top_p=1.0, top_k=-1, stop=None, timeout=None)` | `SampleBatch`: `batch[i][j]` is a `Sample` (`id`, `text`, `finish_reason`, `tokens`); `generate_seconds`, `seconds`. `prompts` is a list, also for one prompt; a prompt is a string, a list of messages, or `{"messages", "tools"}` / `{"tokens"}`. `n` above 1 needs a `temperature` above 0 |
| `live.add_images(images, timeout=1800)` | sends the image files (paths or `data:` URIs) and MP4 video files (paths ending in `.mp4`) the run lacks and waits until it has them; a run started with `images=True` |
| `live.train(batch, rewards, learning_rate=..., advantage="centered", loss="importance_sampling", clip_low=None, clip_high=None, substeps=1, beta1=None, beta2=None, eps=None, weight_decay=None, timeout=None)` | one training call; `rewards` has the batch's shape. Or `episodes=[Episode(samples, reward, group=None)]` for conversations. `TrainStep`, the step's record: `step`, `reward_mean`, `reward_std`, `groups`, `zero_variance_fraction`, `all_best_fraction`, `all_worst_fraction`, `replies`, `completion_length_mean`, `truncated_fraction`, `kl_sample_train`, `step_s`, `generation_s`, `train_s`, `client_s`, `session_s`, `tokens`, `tokens_per_s_per_gpu`, `rows`, `row_tokens`, `row_fill`, `seconds` |
| `live.info()`, `live.finish()`, `live.close()` | the model, context and `round`; save the model and stop the run; stop the keep-alive only |
| `rl.rollouts(id, step=None)` | `RLRollouts`: a few rollouts of a training call (the latest without `step`): `steps`, `step`, `groups` (the best and the worst group: `reward_mean`, `episodes` with `reward`, `advantage`, `turns` of `prompt` / `reply` / `truncated`); CLI `rl rollouts ID [--step N]` |
| `rl.metrics(id)` | `list[RLStep]`: the record of every training call (`TrainStep`'s fields and `ts`), also after the run stopped; CLI `rl metrics ID` |
| `rl.get(id)`, `rl.list()` | `RLRun`: `id`, `state` (`queued`, `starting`, `ready`, `closing`, `closed`, `failed`), `model`, `params`, `ready`, `billable_seconds`, `close_requested`, `checkpoint`, `error`, `done` |
| `rl.wait(id, timeout=None, poll_interval=15, on_poll=None)` | until `ready`; `NexTronError` if it ended instead |
| `rl.env(id)` | `{"TINKER_BASE_URL", "TINKER_API_KEY"}` for the `tinker` SDK; the second holds your key |
| `rl.stop(id, wait=False)` | billing stops, state `closing`; `wait=True` returns at `closed`. The model is saved if it trained a step (then `jobs.wait_checkpoint(id)`, `jobs.download_checkpoint(id, dest)`) |

## Inference

| Call | Returns |
|---|---|
| `inference.estimate(job_id, max_hours)` | `RLEstimate`: `price_per_gpu_hour`, `gpus`, `price_per_hour`, `credits` (held for `max_hours`), `balance`, `enough`; US cents |
| `inference.start(job_id, max_hours, idle_timeout=None, name=None, tags=None)` | `InferenceRun` in `queued` (the fields of `RLRun`; `params.job_id`); 422 for a job that did not succeed or a model without inference, 410 for a job that finished more than 7 days ago |
| `inference.get(id)`, `inference.list()`, `inference.wait(id, timeout=None, poll_interval=15, on_poll=None)`, `inference.stop(id, wait=False)` | as for RL runs |
| `inference.connect(id)` | `LiveRun` with `sample` (`train` raises `NexTronError`) |
| `inference.openai_base_url(id)` | the base URL for an OpenAI client; `api_key` is your NexCloud key |

## CLI

`nextron-cloud` prints JSON and exits 0 on success, 1 on an error (`{"error", "status", "message"}`; `{"error"}` for
one before any request, such as a file that is missing), 3 when `jobs create`, `rl start`, `inference start` or
`sweep` printed estimates instead of starting. `datasets check` exits 1 for a file that does not pass, `datasets
upload` and `import` for a dataset the server finds invalid (it is still printed).

```
nextron-cloud me | models | layouts
nextron-cloud datasets list [--sample] [--table [--model M]] | get ID | check PATH | upload PATH [--name N] | splits URL
nextron-cloud datasets prepare PATH [--out-dir D] [--val-fraction F] [--seed S] [--prompt-column C ... --response-column C | --answer-column C] [--system TEXT]
nextron-cloud datasets import URL [--config C] [--split S] [--name N] | download-url ID | delete ID
nextron-cloud jobs estimate --model M --dataset ID [--layout L] [--param KEY=VALUE ...]
nextron-cloud jobs create   --model M --dataset ID [--layout L] [--param KEY=VALUE ...] [--yes]
                            [--name N] [--tag T ...] [--idempotency-key K]
                            [--wandb-project P [--wandb-run-name R]]   (W&B key from WANDB_API_KEY)
nextron-cloud jobs list [--tag T] | get ID | wait ID | progress ID | billing ID | metrics ID | cancel ID | logs ID [--follow]
nextron-cloud jobs events ID [--after SEQ] [--follow] [--until training|checkpoint]
nextron-cloud jobs serve-rewards ID FILE.py:FUNCTION     # remote-reward GRPO: score until training is over
nextron-cloud jobs wait-checkpoint ID | checkpoint ID | download ID DEST [--include GLOB ...] | export ID DEST
nextron-cloud rl start --model M --max-hours H [--max-context T] [--max-grad-norm G] [--no-grad-clip] [--idle-timeout S] [--images]
                       [--video-fps F] [--video-max-frames N] [--video-frame-tokens T]
                       [--name N] [--tag T ...] [--yes]
nextron-cloud rl list | get ID | wait ID [--timeout S] | env ID | stop ID [--wait]   # env: shell lines for eval, without the key
nextron-cloud rl metrics ID | rollouts ID [--step N]
nextron-cloud inference start --job JOB_ID --max-hours H [--idle-timeout S] [--name N] [--tag T ...] [--yes]
nextron-cloud inference list | get ID | wait ID [--timeout S] | url ID | stop ID [--wait]   # url: an OpenAI client's base URL
nextron-cloud sweep --name N --model M --dataset ID [--layout L] [--param KEY=VALUE ...] --grid KEY=V1,V2 [...] [--yes]
nextron-cloud report [JOB_ID ...] [--tag T] [--table]
```

`--param` values are parsed as JSON when they can be (`max_steps=3`, `learning_rate=1e-4`), else taken as text
(`train_type=lora`). `datasets upload` and `datasets import` wait for the server's check.
