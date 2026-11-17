# Changelog

## 0.1.0rc14

- `jobs.wait_checkpoint` and `jobs.events(until="checkpoint")` end for an RL run that trained no step and so saved no
  checkpoint (`wait_checkpoint` raises `NexTronError`); they waited without end.
- `live.sample` refuses a single prompt given without its list (a string was sampled once per character, a list of
  chat messages was sent as prompts the run could not read), and `n` above 1 at `temperature=0`, before sending.
- `train` on `inference.connect(...)` raises `NexTronError` saying an inference run does not train; it was a 404.
- `datasets upload` sends a file without images as it is; it first packed a copy of it into a temporary tar.
- `datasets prepare` refuses `--prompt-column` without `--response-column` or `--answer-column` (and the reverse), and
  both of those together; it ignored the extra option. A row that lacks a column is left out and counted, not a `KeyError`.
- `datasets prepare` writes image paths that are found from an `--out-dir` reached through a symlink (`/tmp` on macOS).
- `TrainingParams.max_seq_len` takes 1048576 (`kimi-k3.1-pro-1m`); `guide --help` lists every topic.
- Guide serverless-rl: the `tinker` SDK ends its session, and so the run, only at the end of a
  `with tinker.ServiceClient()` block or on `close()`; a script that just returns is billed 3 more minutes.
- `datasets check`, `datasets prepare` and `write_jsonl` take a sample whose text holds `<image>`, `<video>` or
  `<audio>` when the file has no images, as the server does; they refused it. In a file with images the line is
  still an error.
- Guide dpo: Qwen3.8-Flash-Next 180B needs the model's own replies.
- `jobs.billing` reads the bill of an RL or inference run (`price_per_hour`, `seconds_charged`); it raised a
  validation error for a run billed by time.
- A live call's `timeout` also ends a poll whose answer stalls (the request waited out the client's own 60 s).
- `datasets upload` and `datasets import` exit 1 when the server's check finds the dataset invalid; it is still printed.
- `rl start` and `inference start` refuse `--max-hours` and the other numbers at or below zero before asking a price.
- A file that is missing or cannot be read is printed as `{"error": ...}` with exit code 1, not as a traceback.
- Guide inference: the OpenAI example installs `openai` and imports `os`.

## 0.1.0rc13

- A live call's `timeout` also ends its wait through an API outage (the two-minute outage wait ignored it).
- `rl start` and `inference start` without `--yes` say when the model has no such runs, instead of printing an
  empty price and "rerun with --yes".

## 0.1.0rc12

- `datasets prepare` and `write_jsonl` keep samples with images (they were left out with "images go in the user
  messages"); `prepare` rewrites image paths for its output folder.
- `datasets check` reads each image's header as the server does: a file that is not a complete JPEG or PNG, has over
  36 million pixels, or one side over 200 times the other is reported before upload.

## 0.1.0rc11

- Datasets with images: a user message of a `messages` sample can carry JPEG or PNG images as content parts
  (`{"type": "image", "image": "photos/cat.jpg"}`, a path relative to the JSONL file or a base64 `data:` URI; OpenAI's
  `image_url` part is read the same way). `datasets check` reads every image, `datasets upload` sends the samples and
  the image files in one upload, and `Dataset.images` is their number. Qwen3.8-27B and Gemma-4-26B-A4B train on them
  (SFT, full-parameter or LoRA), for organisations images are opened to. Guide: `references/data.md`, "Images".

## 0.1.0rc10

- A `timeout` runs for its whole length: `wait` calls no longer give up one poll interval early (at once when the
  timeout was shorter than the interval), and a live `sample` or `train` poll is no longer than the time left.

## 0.1.0rc9

- Preference training (DPO): `train_type="dpo"` with `beta`, on a dataset of preference pairs (`kind="preference"`):
  `{"messages" | "prompt", "chosen", "rejected"}` or two conversations that differ only in their last reply.
  `data.py` checks and converts them like the server. `nextron-cloud guide dpo`.
- Guides: managed GRPO is reinforcement fine-tuning (RFT).

## 0.1.0rc8

- Inference guide: `stream=True` streams the answer as it is generated (no 100 s limit while it streams); a call
  without it must finish in about 90 seconds.

## 0.1.0rc7

- `LiveRun` keeps calling through a 500 while the API restarts (it repeated only 502-504), so an RL loop survives a
  deploy of the service.
- Inference: a sample keeps a reasoning model's reasoning apart (`Sample.reasoning`, Muse-Glimmer).

## 0.1.0rc6

- Inference: `client.inference` and `nextron-cloud inference start|wait|url|stop|list|get` serve the model of a
  training job that succeeded in the last 7 days, on the GPUs that trained it, billed by time. Call it with
  `client.inference.connect(id).sample(...)` (a batch of prompts in one call) or any OpenAI client
  (`client.inference.openai_base_url(id)`, your NexTron key). `nextron-cloud guide inference`.
- Serverless RL: `qwen3.8-next-180b` (two nodes); the load-time texts say about 20 minutes for it.

## 0.1.0rc5

- Serverless RL: `rl start --no-grad-clip` (`client.rl.start(grad_clip=False)`) starts a run without gradient clipping.

## 0.1.0rc4

- Serverless RL: `client.rl` and `nextron-cloud rl start|wait|env|stop|list|get|metrics|rollouts`: an RL run is a
  loaded model your own loop samples from and trains. `client.rl.connect(id)` drives one with NexTron's own
  calls (`sample` a batch of chat prompts, `train` on a reward per reply or per conversation); it also serves Tinker's
  API for the `tinker` SDK and tinker-cookbook. Every training call returns its record (reward statistics, the time
  split into sampling, training and waiting for your code, how full the trainer's rows were), and
  `rl.metrics(id)` lists them; `rl.rollouts(id)` shows what the model wrote in a call (its best and worst group).
  `nextron-cloud guide serverless-rl`.
- Your own reward for a managed GRPO job (client side only; the service does not take `reward="remote"` yet):
  `jobs.serve_rewards` (CLI `jobs serve-rewards FILE.py:FUNCTION`), `jobs.rollouts`, `jobs.score`; RL samples keep an
  optional `metadata` value for it.

## 0.1.0rc3

- The README's agent prompt covers reinforcement learning: when replies can be checked against known answers it
  weighs GRPO, plans a 2-3 round first run, and reports the reward per round.
- Reinforcement learning: `train_type="grpo"` with the `reward`, `num_generations`, `max_completion_length` and
  `temperature` params; RL samples (prompt + answer) in `check_file`, `write_jsonl` and `datasets prepare --answer-column`;
  `Dataset.kind`; per-round GRPO metrics. Guide: `nextron-cloud guide rl`.

## 0.1.0rc2

- The README's agent prompt asks what you need instead of leaving brackets to fill, and links the source.
- The local format check accepts lines up to 32 MB, as the server now does.
- `datasets list --sample --table --model M`: sample datasets with tokens, longest sample and samples over the
  model's `max_seq_len`. The agent prompt asks whether to use a sample dataset or your own data.
- CLI output of a job includes `done` and `checkpoint_ready`.
- Guide: plan with the steps `estimate` returns, budgets against estimates, `engine_debug_log`, where to get
  `compare_outputs.py`.

## 0.1.0rc1

First release candidate.

- `NexTron` client: `me`, `models`, `layouts`; datasets (upload with a local format check, Hugging Face import,
  wait, download link, delete); jobs (estimate with the server's checks, create with name and tags, wait, progress,
  events, logs, metrics, billing, cancel, checkpoint download with sha256 checks, export).
- Data helpers: `to_messages`, `write_jsonl`, `check_file`, `prepare_file` for messages, ShareGPT, Alpaca, Dolly,
  OpenOrca, prompt/completion, question/answer, problem/solution and input/output rows.
- Experiments: `sweep` (one job per combination, repeat-safe) and `report`.
- `nextron-cloud` CLI, JSON output; `nextron-cloud guide` prints the agent guide shipped in the package.
