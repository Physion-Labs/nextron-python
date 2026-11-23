# HTTP API reference

Everything the SDK and CLI do goes through this API; use it directly from any language. The SDK calls are in
[api.md](api.md); the meaning of each setting is in training.md, data.md, media.md, rl.md, dpo.md, serverless-rl.md and
inference.md.

## Basics

- Base URL: `https://nextron.physionlabs.ai`. Every path below starts with `/v1`.
- Authentication: `Authorization: Bearer nk_live_...` (an API key from the console's **API keys** page; a key belongs to
  one organisation).
- Requests and responses are JSON. Money is in **US cents**, times are **Unix seconds**, sizes are **bytes**.
- Errors: a 4xx or 5xx status with a JSON body `{"error": code, "detail": what to change}`. 401 a missing or wrong key,
  402 not enough balance, 403 not open to your organisation, 404 not found (or not yours), 409 the resource is busy (a
  dataset in use, a checkpoint still uploading), 410 gone (expired), 422 a request the server refuses (bad value, a
  setting the model does not take; the message names it), 429 a limit (10 open jobs, 50 datasets) or too many requests in a
  minute.
- Retries: reads, `POST /v1/training/jobs` with an `Idempotency-Key` header, and cancels are safe to retry on
  connection errors and 429 / 502 / 503 / 504. The same `Idempotency-Key` with the same body returns the same job.

## Account

| call | returns |
|---|---|
| `GET /v1/me` | `org` (`id`, `name`, `role`, `credits`: balance), `orgs` (every organisation of a signed-in user), `billing` (`prices`: `{model: {method: cents per million tokens}}`, `checkout`, `min_cents`, `max_cents`) |
| `GET /v1/models` | every model a job can train, live (below) |
| `GET /v1/layouts` | every GPU setup: `id`, `name`, `nodes`, `gpus`, `models`, `tokens_per_s` (per GPU, per model), `available`, `queued`, `coming_soon` |
| `GET /v1/billing/usage` | this month's charges by day: `month_start`, `days` (`[{day, credits}]`), `held` (credits held for running jobs) |
| `GET /v1/billing/ledger` | every balance change (top-ups, grants, charges) |

`GET /v1/models` returns, per model: `id`, `name`, `maker`, `params`, `active_params`, `architecture`, `input` (`text`,
and `image` / `video` where they are open to your organisation), `tools` (whether it trains on tool calls), `batch_size` (whether `params.batch_size` is taken),
`validation_set`, `layouts` (`{id: {gpus, tokens_per_s_per_gpu, available, queued}}`) and `methods`:

| method | fields |
|---|---|
| `full`, `lora` | `max_seq_len`, `layouts`, `price_per_mtok`, `defaults` (the optimizer and step settings a job trains with unless it sets them: `max_grad_norm` (0 = no clipping), `adam_beta1`, `adam_beta2`, `adam_epsilon`, and for `lora` `lora_rank`, `lora_alpha`, `lora_dropout`), `max_tokens_per_gpu` (the largest `tokens_per_gpu`; null where the model trains with its tuned step) |
| `dpo` | as above, plus `learning_rate` (the model's default) |
| `grpo` | `max_seq_len` (prompt + completion), `max_prompt_len`, `max_completion_length`, `rewards` |
| `interactive` (Serverless RL), `inference` | `max_seq_len` (a prompt and its reply), `layouts`, `price_per_gpu_hour` |

## Datasets

| call | does |
|---|---|
| `POST /v1/datasets` `{"name", "bytes", "images": false}` | reserves an upload. Returns the dataset (`state: "uploading"`) with `upload_url` and `upload_headers` (one `PUT` of the file), or `part_urls` for a large file (multipart, then `complete`). `images: true` means the upload is a bundle (a tar of the samples and their image and video files, as the SDK writes it) |
| `POST /v1/datasets/{id}/complete` | after the upload: the server's check starts (`checking`, then `ready` or `invalid`) |
| `GET /v1/datasets`, `GET /v1/datasets/{id}` | your datasets and the sample datasets: `id`, `name`, `state`, `samples`, `est_tokens`, `too_long`, `tokens` (`{model: {total, max, over: {max_seq_len: count}}}`), `errors`, `kind` (`chat`, `rl`, `preference`, `images`), `images`, `videos`, `tools`, `sample`, `validation`, `source`, `converted_from` |
| `DELETE /v1/datasets/{id}` | 409 while an unfinished job uses it |
| `GET /v1/datasets/{id}/download` | a one-hour link to the file as training reads it |
| `GET /v1/datasets/{id}/plan?model=&layout=&max_seq_len=&batch_size=&tokens_per_gpu=&gradient_accumulation_steps=&train_type=` | the batch a job would train with: `layout`, `gpus`, `batch` (samples a step), `pack_len`, `tokens_per_gpu`, `micro_steps`, `steps_per_epoch`, `min_batch`, `efficient`, `fits` |
| `POST /v1/datasets/huggingface/splits` `{"url", "hf_token"?}` | `[{"config", "split", "rows", "bytes"}]` |
| `POST /v1/datasets/huggingface` `{"url", "config", "split", "name"?, "hf_token"?}` | imports one split (`importing`, then `checking`) |

The file format is in data.md, images and videos in media.md. Limits: 50 datasets, 5 GB a file, 20 GB a bundle.

## Training jobs

`POST /v1/training/estimate` and `POST /v1/training/plan` take the same body as `POST /v1/training/jobs` and start
nothing: `estimate` returns the price (`tokens`, `price_per_mtok`, `credits` held, `balance`, `enough`, `metered`,
`params`, `steps`, `truncated_samples`, `warnings`, `video`); `plan` returns the checks only (`params`, `steps`,
`truncated_samples`, `warnings`). Both refuse (422) what `jobs` would refuse.

`POST /v1/training/jobs` (header `Idempotency-Key` optional):

| field | |
|---|---|
| `model` | a model id from `GET /v1/models` |
| `dataset_id` | a `ready` dataset of the method's kind |
| `layout` | a GPU setup id, or `"auto"`: the smallest free one that trains the model (a larger setup picked by hand costs 5% more per size step) |
| `params` | the training settings, below |
| `wandb` | optional `{"api_key", "project", "entity"?, "run_name"?}`: also log to your Weights & Biases (the key is stored encrypted, never returned) |
| `name`, `tags` | your label (up to 200 characters) and up to 20 tags of `A-Z a-z 0-9 _ . : = / + -` |

### `params`

All optional. "Model's" means the model's template: `GET /v1/models` shows it under `methods[...].defaults`.

| key | default | allowed | method |
|---|---|---|---|
| `train_type` | `"full"` | `full`, `lora`, `grpo`, `dpo` | |
| `learning_rate` | 1e-5 (grpo 1e-6, dpo the model's) | > 0, up to 1e-3 | all |
| `num_train_epochs` | 1 | 1-20 | full, lora, dpo |
| `max_steps` | | 1-100,000; with epochs, whichever comes first; required for grpo (rounds) | all |
| `lr_scheduler_type` | `"cosine"` | `cosine`, `linear`, `constant` | full, lora, dpo |
| `warmup_ratio` | 0.03 | 0-0.5 | full, lora, dpo |
| `weight_decay` | 0.1 | 0-1 | full, lora, dpo |
| `max_seq_len` | the method's longest | 32768, 65536, 131072, 262144, 1048576, up to the method's | full, lora |
| `seed` | 42 | 0 to 2^31-1 | all |
| `max_grad_norm` | model's | 0 (no clipping) to 1e6 | all |
| `adam_beta1`, `adam_beta2` | model's (0.9, 0.999 unless the model sets them) | 0 to < 1 | all |
| `adam_epsilon` | model's (1e-8) | > 0, up to 1e-3 | all |
| `batch_size` | each GPU packs `max_tokens_per_gpu` tokens | samples a step (grpo: prompts a round, up to 256), multiple of 8 and of the GPUs, 8-8,192; where `batch_size` is true in `/v1/models` | full, lora, grpo |
| `tokens_per_gpu` | model's (`max_tokens_per_gpu`) | each GPU's packed row a micro-step: 8,192 up to `max_tokens_per_gpu`, at least the dataset's longest sample; not with `batch_size` | full, lora, dpo (where `max_tokens_per_gpu` is set) |
| `gradient_accumulation_steps` | 1 | 1-64 micro-steps an optimizer step; not with `batch_size` (which is split into micro-steps itself) | full, lora, dpo (as above) |
| `lora_rank` | 32 | 1-128 | lora |
| `lora_alpha` | 64 | 1-512 | lora |
| `lora_dropout` | model's | 0-0.5 | lora |
| `val_dataset_id` | | a ready dataset of at most 5,000 samples, text only | full, lora |
| `eval_steps` | about 10 evaluations | steps between validation passes | full, lora |
| `video_fps`, `video_max_frames`, `video_frame_tokens` | the model's | media.md | full, lora, grpo |
| `reward` | | `boxed`, `exact`; required | grpo |
| `num_generations` | 8 | 2-16 | grpo |
| `max_completion_length` | 8192 | 256-8192 | grpo |
| `temperature` | 1.0 | > 0, up to 2 | grpo |
| `beta` | 0.1 | > 0, up to 1 | dpo |
| `engine_debug_log` | false | also log the engine's setup | all |

A step trains `gpus x tokens_per_gpu x gradient_accumulation_steps` tokens (rows are packed with whole samples, so a
little less), or `batch_size` samples. A setting a method does not take, or a value the model does not allow, is
refused with the reason.

### Reading a job

| call | returns |
|---|---|
| `GET /v1/training/jobs` | your organisation's jobs, newest first |
| `GET /v1/training/jobs/{id}` | `id`, `name`, `tags`, `state` (`queued`, `submitting`, `pending`, `running`, `succeeded`, `failed`, `cancelled`), `model`, `layout`, `dataset_id`, `params`, `metrics`, `checkpoint`, `error`, `done`, `checkpoint_ready`, `timeline`, `notices`, `tokens_trained`, `config` |
| `GET /v1/training/jobs/{id}/progress` | `phase`, `jobs_ahead`, `training` (step, ETA), `checkpoint` (stage, bytes), `timeline`, `last_activity`, `idle_seconds` |
| `GET /v1/training/jobs/{id}/events?after=&limit=` | `[{seq, ts, type, data}]`: `job.state`, `training.step`, `training.validation`, `checkpoint.stage`, ...; poll with `after` = the last `seq` |
| `GET /v1/training/jobs/{id}/metrics` | per step: `step`, `total_steps`, `loss`, `val_loss`, `lr`, `step_time_s`, `tokens_per_s_per_gpu`, `tokens_per_s_total`, `samples`, `real_tokens`, `trained_tokens`, `padding_tokens`, `ts` (grpo adds the reward fields) |
| `GET /v1/training/jobs/{id}/logs?cursor=&limit=` | `{lines: [{seq, ts, line}], next_cursor}`; pass `next_cursor` back as `cursor` |
| `GET /v1/training/jobs/{id}/billing` | `billing` (`held`, `settled`, `released`, `not_metered`), `held_credits`, `charged_credits`, `price_per_mtok`, `tokens_charged`, `settled_at` |
| `POST /v1/training/jobs/{id}/cancel` | queued: cancelled now; running: stops soon, no checkpoint; charged for the tokens trained |
| `GET /v1/training/jobs/{id}/checkpoint` | `step`, `bytes`, `expires`, `files` (`name`, `bytes`, `sha256`, `url`: a 24-hour link). 409 while uploading, 410 after 30 days |
| `GET /v1/training/jobs/{id}/manifest`, `.../export`, `.../trace`, `.../telemetry` | the sha256 manifest, the job record as a zip, a per-rank Chrome trace, GPU and host utilisation |
| `GET /v1/training/jobs/{id}/rollouts`, `POST .../rollouts/{batch_id}/scores` `{"scores": [...]}` | remote-reward grpo: the batch waiting for scores, and its scores (rl.md) |

## Serverless RL runs

| call | does |
|---|---|
| `POST /v1/training/sessions/estimate` `{"model", "max_hours"}` | `price_per_gpu_hour`, `gpus`, `price_per_hour`, `credits` held, `balance`, `enough` |
| `POST /v1/training/sessions` | starts a run: `model`, `max_hours` (required), `max_context` (2,048-16,384), `max_grad_norm`, `grad_clip`, `idle_timeout` (60-14,400 s), `images`, `video_fps` / `video_max_frames` / `video_frame_tokens`, `name`, `tags` |
| `GET /v1/training/sessions`, `GET /v1/training/sessions/{id}` | `id`, `state` (`queued`, `starting`, `ready`, `closing`, `closed`, `failed`), `model`, `params`, `ready`, `billable_seconds`, `close_requested`, `checkpoint`, `error`, `images` |
| `POST /v1/training/sessions/{id}/close` | billing stops; the model is saved if it trained a step |
| `POST /v1/training/sessions/{id}/images` `{"dataset_id"}` | gives a run started with `images` a dataset's images and videos (named `sha256:<hex>` in prompts) |
| `GET /v1/training/sessions/{id}/rollouts?step=` | the best and worst group of a training call |
| `/v1/training/sessions/{id}/nextron/v1/...` | the run's own calls: `sample`, `batches/{id}`, `train`, `steps/{id}`, `info`, `heartbeat`, `images/missing`, `finish` (serverless-rl.md) |
| `/v1/training/sessions/{id}/tinker/...` | Tinker's HTTP API, for the `tinker` SDK |

## Inference runs

| call | does |
|---|---|
| `POST /v1/inference/estimate` `{"job_id", "max_hours"}` | the price, as for an RL run |
| `POST /v1/inference` `{"job_id", "max_hours", "idle_timeout"?, "images"?, "name"?, "tags"?}` | serves a job's model (a job that succeeded in the last 7 days) |
| `GET /v1/inference`, `GET /v1/inference/{id}`, `POST /v1/inference/{id}/stop` | as for RL runs |
| `POST /v1/inference/{id}/images` `{"dataset_id"}` | images and videos for a run started with `images` |
| `/v1/inference/{id}/v1/chat/completions`, `.../v1/completions`, `.../v1/models` | OpenAI's API: the base URL is `/v1/inference/{id}/v1`, the key your NexCloud key |
| `/v1/inference/{id}/nextron/v1/...` | `sample` and `batches/{id}` for a whole batch of prompts |

## API keys

| call | does |
|---|---|
| `GET /v1/keys` | your organisation's keys (never the secret) |
| `POST /v1/keys` `{"name"}` | a new key; its secret is returned once. Organisation owners only |
| `DELETE /v1/keys/{id}` | revokes a key |
