# Data

## Your data or a sample dataset

Ask the owner which. Their own data (a file or a Hugging Face link) is what teaches the model their task; the rest of
this page is about preparing it. Sample datasets, which every account can train on, are for trying the service or a
model without preparing data:

```bash
nextron-cloud datasets list --sample --table --model qwen3.8-27b
```

lists each one's samples, tokens and longest sample for that model, and how many samples are over the model's
`max_seq_len` (they would be truncated: pick the model's long-context id, such as `...-256k`, or accept it).
`source` says where each comes from. The ones marked `validation` are held-out sets: pass one as `val_dataset_id`
with the training set of the same name. Sample datasets hold tens of millions of tokens, so a full epoch is costly:
cap a trial with `max_steps` and show the owner the estimate.

## The format

A dataset is a JSONL file: UTF-8, one JSON object per line, at most 32 MB a line and 5 GB a file.

```json
{"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]}
```

- Roles are `system`, `user`, `assistant`. At most one system message, and only first.
- At least one non-empty assistant message.
- `content` is a string, or OpenAI content parts: `{"type": "text", "text": ...}`, and in user messages image and
  video parts (see [Images and videos](#images-and-videos)).
- Audio is refused, and so is an `image(s)` / `video(s)` / `audio(s)` field beside the messages. Images and videos go
  inside user messages: see [Images and videos](#images-and-videos).

These are converted to `messages` (by `to_messages` / `write_jsonl` locally, and by the server on upload):

| format | sample |
|---|---|
| ShareGPT, LMSYS | `{"conversations" \| "conversation": [{"from": "human" \| "gpt" \| "system", "value": ...}]}`, or turns with `role`/`content` |
| Alpaca, Dolly | `{"instruction": ..., "input" \| "context": ..., "output" \| "response": ...}` (instruction and input/context joined by a blank line; other fields such as `category` ignored) |
| prompt / completion | `{"prompt": ..., "completion": ...}` or `{"prompt": ..., "response": ...}` |
| question / answer | `{"query" \| "question": ..., "response" \| "answer": ...}` (e.g. OpenOrca) |
| problem / solution | `{"problem": ..., "solution": ...}` (MATH, NuminaMath, Magicoder), `{"input": ..., "output": ...}` |

A top-level `"system"` or `"system_prompt"` string becomes the system message. Hugging Face datasets in these formats
import directly (`import_huggingface`), no conversion needed. Values must be strings (an `"answer": 7` is refused:
write `"7"`).

## What training learns from

Checked against the training template of Qwen3.8-27B; the other models follow the same rules.

- These rules were checked against Qwen3.8-27B's template. Whether another model has a thinking mode shows in its
  chat template (`chat_template.jinja` in a checkpoint): no `<think>` there, no think block.
- **Every assistant message is trained**, in every turn of a multi-turn sample, including its end-of-turn token.
  System and user messages are context only.
- **Reasoning:** jobs train in non-thinking mode. An assistant message without reasoning is trained as
  `<think>\n\n</think>\n\n` + the answer, so the model learns to answer directly. To train reasoning, put it in the
  assistant content as `<think>\n...\n</think>\n\n` followed by the answer; it is kept and trained. Serve the
  checkpoint the same way (see checkpoints.md).
- **Long samples:** a sample longer than the job's `max_seq_len` keeps its **last** `max_seq_len` tokens (the start
  of the conversation is cut). The job's notices say how many were cut.
- Several samples are packed into one row for speed; attention stays within each sample.

## Build and check locally

From a file (`.csv`, `.jsonl`, `.json`) to `train.jsonl` + `val.jsonl`, shuffled, unusable rows left out and counted:

```bash
nextron-cloud datasets prepare tickets.csv --out-dir data --prompt-column ticket --response-column reply \
    --system "You are our support agent." --val-fraction 0.1
```

Repeat `--prompt-column` to join several columns into the user message (blank line between, empty ones skipped), e.g.
`--prompt-column question --prompt-column context`.

(`prepare_file(...)` in `nextron_cloud.data` does the same from Python.) Without the column options, each row must
already be in one of the formats above. Read `skipped` and `errors` in its output before uploading.

For anything else, build the rows yourself:

```python
from nextron_cloud import check_file, to_messages, write_jsonl

messages = to_messages({"question": "2+2?", "answer": "4"})     # one sample; ValueError says what is wrong
count = write_jsonl(rows, "train.jsonl")                         # any accepted format in, "messages" JSONL out
result = check_file("train.jsonl")                               # the server's check: samples, chars, errors
```

- `write_jsonl` raises on the first bad sample with its number. To skip bad rows instead, call `to_messages` per
  row and catch `ValueError`.
- `check_file` reports the first five errors with line numbers, like the server.
- pandas: `write_jsonl(df.to_dict("records"), path)` when the columns are one of the formats above; otherwise build
  `{"messages": [...]}` per row, as in this CSV example (columns `ticket`, `reply`):

```python
import csv, random
from nextron_cloud import write_jsonl

with open("tickets.csv", newline="", encoding="utf-8") as f:
    rows = [{"messages": [{"role": "system", "content": "You are our support agent."},
                          {"role": "user", "content": r["ticket"]},
                          {"role": "assistant", "content": r["reply"]}]} for r in csv.DictReader(f)]
random.Random(0).shuffle(rows)
cut = max(1, len(rows) // 10)
write_jsonl(rows[cut:], "train.jsonl")
write_jsonl(rows[:cut], "val.jsonl")
```

- Hugging Face `datasets`: `write_jsonl(ds, path)` works the same way (rows are dicts).
- Shuffle before splitting off a validation set, so both come from the same distribution.

## Upload or import

```python
dataset = client.datasets.upload("train.jsonl", name="support-v3")   # local check, then upload; state "checking"
dataset = client.datasets.wait(dataset.id)                           # "ready", or "invalid" with dataset.errors

client.datasets.huggingface_splits("org/name")                       # configs, splits, rows, bytes
dataset = client.datasets.wait(client.datasets.import_huggingface("org/name", config="default", split="train").id)
```

Hugging Face imports accept a link or `org/name`; gated or private ones need `hf_token=`. The split must already be in
one of the formats above; otherwise download it, convert with `write_jsonl` and upload.

## What the server reports

After its check, `dataset` has:

- `samples`, `est_tokens` (the largest token count over the models; the estimate uses it), `too_long` (samples over
  65,536 tokens for at least one model).
- `tokens[model]`: `total` tokens, `max` (the longest sample), and `over`: how many samples are longer than each
  `max_seq_len` a job can pick (32768, 65536, 131072, 262144, 1048576; a dataset checked before a length was added
  has no count for it). Compare `max` and `over` with the model's `max_seq_len` from `client.models()` to see what
  would be truncated.
- `converted_from`: the format it was converted from, if any. `datasets.download_url(id)` gives the file exactly as
  training reads it.

Any ready dataset of at most 5,000 samples can be a job's `val_dataset_id`; upload it like the training set. The
`validation` flag only marks the validation sets among the sample datasets and is false on your own uploads, which is
fine.

Limits: 50 datasets an account (delete old ones with `datasets.delete`), validation sets at most 5,000 samples. Sample
datasets (`sample=True` in `datasets.list()`) can be trained on by every account and cannot be deleted.

## Images and videos

Qwen3.8-27B, Gemma-4-26B-A4B, Qwen3.8-Next-180B and Muse-Glimmer-30B train on images, and the first three on videos
too (SFT, full-parameter or LoRA, and their long-context ids such as `qwen3.8-27b-256k`). Images and videos are opened
organisation by organisation, each on its own: if `datasets upload` answers 403, images are not open to yours yet; a
dataset whose videos are not open to yours ends `invalid` with "videos are not open to this organisation yet".
`nextron-cloud models` lists `image` / `video` in a model's `input` only where they are open to you.

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "photos/cat.jpg"}, {"type": "text", "text": "What is this?"}]},
              {"role": "assistant", "content": "A cat."}]}
{"messages": [{"role": "user", "content": [{"type": "video", "video": "clips/race.mp4"}, {"type": "text", "text": "Who wins?"}]},
              {"role": "assistant", "content": "The red car."}]}
```

- An image is a file path relative to the JSONL file, or a base64 `data:` URI. OpenAI's
  `{"type": "image_url", "image_url": {"url": ...}}` is read the same way. A video is a file path. A URL is not
  fetched: download the file first.
- Images: JPEG and PNG only, up to 20 MB and 36 million pixels each, no side over 200 times the other, at most 16 a
  sample. Videos: MP4 only, up to 500 MB, 1 to 60 seconds and 3840 x 2160 a frame each, at most 4 a sample.
  In user messages only.
- Do not write `<image>`, `<video>` or `<audio>` in the text of such a dataset.
- `datasets check` reads every image file's header and every video's MP4 boxes, as the server does on upload: type,
  size, pixel count, length. The server then decodes each file; one whose data is broken behind a good header makes
  the dataset `invalid` there.
- `datasets prepare` and `write_jsonl` keep a sample with images or videos as it is (`prepare` rewrites their paths
  for its output folder).
- `datasets upload` packs the samples and the image and video files into one upload
  (each file once, up to 20 GB together). Nothing else changes: `wait`, then train.
- An image counts as tokens, and is billed as them: up to 514 on Qwen3.8-27B (a 32-pixel square each, so a small image
  costs less), about 260-282 on Gemma-4-26B-A4B whatever its size. A video counts as the tokens of the frames
  training reads (see below). `tokens` on the dataset includes them per model.
- A sample with images or videos is never cut: a job is refused if the dataset's longest sample is over `max_seq_len`.
- Training reads a copy of each image: 8-bit RGB, transparency on white, turned as its EXIF says, and scaled down
  only above 1.6 million pixels. Each video is re-encoded as H.264 at up to 16 frames a second and up to 1280 x 720,
  with at least 32 frames; its audio is dropped.
- Not with images or videos yet: validation sets and DPO. A sample with images or videos must fit one GPU's row (128K
  tokens on Qwen3.8-27B, 64K on the others), also on a long-context id; text samples beside it may be longer.

### Video settings

A job (`--param`) or an RL run (`rl.start(...)`, `rl start --video-...`) chooses how its videos are sampled:

| param | Qwen3.8-27B, Qwen3.8-Next-180B | Gemma-4-26B-A4B |
|---|---|---|
| `video_fps` | 1, 2, 4 (default), 8, 16 frames a second | - (frames are spread over the video) |
| `video_max_frames` | 2 to 960, default 40 | 1 to 256, default 32 |
| `video_frame_tokens` | 256, 512 (default), 900 (a 720p frame pair in full) | 70 (default), 140, 280 |

```bash
nextron-cloud jobs create --model qwen3.8-27b --dataset ds_... --param video_fps=8 --param video_max_frames=200
```

- Gemma: at most 20,480 tokens a video (frames x (tokens a frame + 9)): 256 frames at 70, 137 at 140, 70 at 280.
  Gemma RL runs keep the defaults (32 x 70).
- More frames and tokens train more detail and cost more: a video's tokens are about its frames x its tokens a frame
  (Qwen: a pair of frames shares one grid). `jobs create` without `--yes` shows the settings (`video`) and the
  dataset's tokens at them.
- A choice the model does not take, or a video over the row at the chosen settings, is refused with what fits.
- Gemma will take at most 20,480 tokens a video (256 frames at 70, about 70 at 280). Its RL runs and RFT keep 32
  frames at 70 tokens.
  Serverless RL takes them in its prompts: see serverless-rl.md.
