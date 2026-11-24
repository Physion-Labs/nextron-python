# Changelog

## 0.1.0rc32

- A dataset file can be up to 10 GB (was 5 GB).

## 0.1.0rc31

- `Dataset.estimated`: a file over 50 MB is ready with token counts from a sample; a model's first job measures every
  sample before it starts.

## 0.1.0rc30

- `jobs.wait_checkpoint` raises when the checkpoint was deleted before its upload, instead of waiting on.
- Guide: a run's own `images of ...` dataset is deleted once the run has ended.

## 0.1.0rc29

- Guide: an inference run serves its model's training length (65,536; 131,072 on `muse-glimmer-30b`), not 16K.

## 0.1.0rc28

- `datasets.upload_images(paths)` checks an evaluation set's images and videos before an inference or RL run starts;
  `live.add_images(paths, dataset_id=...)` gives them to the run without another upload (inference.md).
- `ModelInfo.tools` and `Dataset.videos` / `tools` / `converted_from` are typed fields.
- Guide: the small Qwen models (`qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-9b-base`, `qwen3.6-35b-a3b`,
  `qwen3.5-35b-a3b-base`) in media.md, their time and checkpoint sizes, image prompts on inference runs, and the
  inference context limit for videos.

## 0.1.0rc27

- Tool calls in chat datasets, in OpenAI's format (`tools`, assistant `tool_calls`, `tool` messages). Ai2/Dolci,
  ShareGPT (`function_call` / `observation`) and Hermes (`<tool_call>`) rows are converted to it; `nextron-cloud
  models` shows which models take tools (data.md, "Tool calls").

## 0.1.0rc26

- Training settings: `max_grad_norm` (0 = no clipping), `adam_beta1`, `adam_beta2`, `adam_epsilon`, `tokens_per_gpu`,
  `gradient_accumulation_steps`, and for LoRA `lora_rank`, `lora_alpha`, `lora_dropout` (training.md).
  `client.models()` shows each method's `defaults` and `max_tokens_per_gpu`.
- Guide: a full HTTP API reference (`nextron-cloud guide http-api`): every endpoint, field and setting.

## 0.1.0rc25

- Media guide: hosted Qwen and Gemma video settings and verified evaluation results.

## 0.1.0rc24

- Hosted inference supports reusable media uploads with `images=True` and `live.add_images`.

## 0.1.0rc23

- Guide: images and videos have their own page (`nextron-cloud guide media`): models, format, limits, how each video
  setting samples frames, tokens and cost, converting with ffmpeg, an end-to-end example, and serving a video model
  with the frames it trained on. Videos in Serverless RL prompts are 1 to 60 seconds (not 10 minutes).
- `jobs.estimate` returns `video`: the video settings the job would train with.
- Guide: shorter cost section in training.md; repeated sentences removed.

## 0.1.0rc22

- Guide: Gemma-4-26B-A4B jobs take `video_max_frames` 1 to 256 and `video_frame_tokens` 70, 140 or 280 (at most
  20,480 tokens a video); its RL runs keep 32 x 70.

## 0.1.0rc21

- Video settings: `video_fps`, `video_max_frames` and `video_frame_tokens` on jobs (`--param`) and RL runs
  (`rl.start`, `rl start --video-fps / --video-max-frames / --video-frame-tokens`) on the Qwen models; the guide lists the
  choices (Gemma at its defaults for now).
  Videos are 1 to 60 seconds, cleaned at up to 16 frames a second.

## 0.1.0rc20

- Guide: Qwen3.8-27B packs about 128K tokens per GPU a step by default; the long-context ids have no LoRA; DPO's
  default learning rate is the model's (`methods["dpo"].learning_rate`); `rl start --images` in the CLI reference.

## 0.1.0rc19

- `nextron-cloud rl start --images`: a run whose prompts hold images and videos, from the command line.
- Guide: videos are opened per organisation apart from images; a dataset with videos an organisation cannot use ends
  `invalid` with "videos are not open to this organisation yet".

## 0.1.0rc18

- Serverless RL runs with images: `rl.start(..., images=True)`; a prompt's user message may hold image and video
  parts with local paths, which `sample` sends to the run first (`live.add_images` sends a set ahead of the loop).
- Videos (MP4) in datasets and Serverless RL prompts: `{"type": "video", "video": "clip.mp4"}` in a user message, up to
  4 a sample, checked locally from the MP4 boxes, packed into the upload as images are; `CheckResult.videos` counts
  them and `live.add_images` takes `.mp4` files.
- RFT datasets (prompts with an answer) may hold images and videos in their user messages, as chat samples do.
- The long-context model ids of Qwen3.8-27B, Gemma-4-26B-A4B and Qwen3.8-Next-180B take datasets with images and
  videos (guide).

## 0.1.0rc17

- Guide: LoRA is on every model but `kimi-k3.1-pro`, and the larger models train it on fewer GPUs than a full
  fine-tune.

## 0.1.0rc16

- The product is called NexCloud: the guides, the CLI's help and the docstrings say so. The package, the
  `nextron-cloud` command, `NEXTRON_API_KEY` and the `NexTron` client keep their names.

## 0.1.0rc15

- `datasets prepare` with column options leaves out and counts a row that lacks one of the named columns (it kept the
  row without that part of the prompt) and a row that is not an object (it stopped with a traceback).
- An RL or preference sample whose prompt (its last user message) is empty or only whitespace does not pass
  `datasets check`, as on the server.
- In a file with images, `datasets check` refuses `<image>`, `<video>` or `<audio>` written in any message, the system
  prompt and every sample format, as the trainer reads them all; it checked only the text of user messages.
- `datasets check` holds a file with images to a bundle's limits (its samples without the images up to 5 GB, with them
  up to 20 GB, at most 200,000 different images); it refused such a file over 5 GB, data URIs included.
- `datasets prepare` reads each image and leaves out a row whose image cannot be used, and in a file with images a row
  without images that has a media tag in its text; `write_jsonl` raises on both. They passed and the check failed later.

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
