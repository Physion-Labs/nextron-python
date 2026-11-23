# Images and videos

| model | images | videos |
|---|---|---|
| `qwen3.8-27b`, `qwen3.8-next-180b` (Qwen3.8-Flash-Next), `gemma-4-26b-a4b`, and their `...-256k` ids | yes | yes |
| `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-9b-base`, `qwen3.6-35b-a3b`, `qwen3.5-35b-a3b-base` | yes | yes |
| `muse-glimmer-30b` | yes | not yet |

They train with SFT (full or LoRA) and RFT, and go in Serverless RL prompts (serverless-rl.md). Not yet: DPO and
validation sets. Images and videos are opened organisation by organisation, each on its own. `nextron-cloud models`
lists `image` / `video` in a model's `input` only where they are open to yours:

- images not open: `datasets upload` answers 403;
- videos not open: the dataset ends `invalid` with "videos are not open to this organisation yet".

## The format

An image or a video is a content part of a user message:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "photos/cat.jpg"}, {"type": "text", "text": "What is this?"}]},
              {"role": "assistant", "content": "A cat."}]}
{"messages": [{"role": "user", "content": [{"type": "video", "video": "clips/race.mp4"}, {"type": "text", "text": "Who wins?"}]},
              {"role": "assistant", "content": "The red car."}]}
```

- A file path relative to the JSONL file. An image may also be a base64 `data:` URI, or OpenAI's
  `{"type": "image_url", "image_url": {"url": ...}}`. URLs are not fetched: download the file first.
- In user messages only. Do not write `<image>`, `<video>` or `<audio>` in the text.
- Audio is refused, and so is an `image(s)` / `video(s)` / `audio(s)` field beside the messages.

| | files | each file | a sample |
|---|---|---|---|
| images | JPEG, PNG | up to 20 MB and 36 million pixels; no side over 200 times the other | up to 16 |
| videos | MP4 | up to 500 MB, 1 to 60 seconds, 3840 x 2160 a frame | up to 4 |

Other formats, and longer videos, convert with ffmpeg:

```bash
ffmpeg -i clip.mov -c:v libx264 -pix_fmt yuv420p -an clip.mp4                              # to MP4 (H.264)
ffmpeg -i long.mp4 -c:v libx264 -an -f segment -segment_time 60 -reset_timestamps 1 part_%03d.mp4   # 60 s parts
```

## End to end

```bash
nextron-cloud datasets check data/train.jsonl     # reads every image header and every video's MP4 boxes, locally
nextron-cloud datasets upload data/train.jsonl    # packs the JSONL and every file it names into one upload
nextron-cloud jobs create --model qwen3.8-27b --dataset ds_... --param video_fps=2 --param video_max_frames=60
# -> the estimate, with the video settings (`video`) and the dataset's tokens at them; add --yes to start
```

- `upload` sends each file once, up to 20 GB together. The server then decodes every file; one whose data is broken
  behind a good header makes the dataset `invalid`.
- `datasets prepare` and `write_jsonl` keep samples with images or videos as they are (`prepare` rewrites the paths
  for its output folder).
- A runnable example, data to trained model:
  [nextron-cookbook's `video_sft`](https://github.com/Physion-Labs/nextron-cookbook/tree/main/recipes/video_sft).

## What training reads

- Images: 8-bit RGB, transparency on white, turned as their EXIF says, scaled down only above 1.6 million pixels.
- Videos: re-encoded as H.264 at up to 16 frames a second and up to 1280 x 720; the audio is dropped. Training then
  samples frames from each, by the job's video settings (below).
- A sample with images or videos is never cut: a job is refused if the longest one is over `max_seq_len`. It must also
  fit one GPU's row: 128K tokens on `qwen3.8-27b` and `gemma-4-26b-a4b`, 64K on the others, also on a long-context id.
  Text samples beside it may be longer.

## Video settings

A job (`--param`) or an RL run (`rl.start(...)`) chooses how its videos are sampled:

| param | Qwen models | `gemma-4-26b-a4b` |
|---|---|---|
| `video_fps` | 1, 2, 4 (default), 8, 16 | - |
| `video_max_frames` | 2 to 960, default 40 | 1 to 256, default 32 |
| `video_frame_tokens` | 256, 512 (default), 900 | 70 (default), 140, 280 |

- **Qwen** reads `video_fps` frames a second, but at most `video_max_frames`, spread evenly over the whole video. At
  the defaults a 10-second video gets 40 frames (4 a second) and a 60-second one also 40 (about 0.7 a second). Two
  frames share one grid of up to `video_frame_tokens` tokens; 900 holds a 720p pair at full detail.
- **Gemma** always reads `video_max_frames` frames spread evenly over the video, each up to `video_frame_tokens`
  tokens, and at most 20,480 tokens a video: 256 frames at 70, 137 at 140, 70 at 280. Its RL runs and RFT keep the
  defaults (32 x 70).
- Fast motion: raise `video_fps`. Long videos: raise `video_max_frames` so the frames stay close together. Small text
  or detail in the picture: raise `video_frame_tokens`. Each costs tokens.
- A choice the model does not take, or a video that no longer fits the row, is refused with what fits.
- An inference run serves the `inference` method's `max_seq_len` a request, video tokens included: the model's
  training length (65,536; 131,072 on `muse-glimmer-30b`), so a request as long as a training sample fits. A longer
  one is refused with HTTP 400 naming both lengths.

## Tokens and cost

Images and videos count as tokens, and are billed as them. `dataset.tokens[model]` includes them at the default
settings; the estimate counts them at the job's.

| | tokens |
|---|---|
| an image, Qwen models | up to 514 (one per 32 x 32 pixels, so a small image costs less) |
| an image, `gemma-4-26b-a4b` | about 260-282, whatever its size |
| a 720p video of 30 s, Qwen defaults (40 frames, 512) | about 9,800 |
| the same at `video_fps=2`, `video_max_frames=120`, `video_frame_tokens=900` | about 53,000 |
| a 320 x 240 video of 8 s, Qwen defaults | about 2,400 |
| a video, Gemma defaults (32 frames, 70) | about 2,500 |

So 1,000 such 30-second clips at the Qwen defaults are about 10M tokens an epoch: about $10 on `qwen3.8-27b` at $1.00
per million tokens (take the price from `nextron-cloud models`).

## Serve the trained model

Start an inference run from the completed job with `images=True` (CLI `inference start ... --images`) and wait until
it is ready ([inference.md](inference.md)). Then:

- `connect(...).sample(...)`: images and videos as content parts, as in a dataset (`{"type": "image", "image":
  "clock.png"}`, `{"type": "video", "video": "race.mp4"}`); the files are sent to the run first.
- OpenAI base URL: an image as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`, an MP4 as
  `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`, followed by the text question.

Stop the run after evaluation.

A run serves videos with the frame settings the job trained with, read from the checkpoint:

- Qwen models: the training frame rate, frame cap and pixel limits. In the bounce-counting recipe on `qwen3.8-27b`,
  20 clips matched the training reference's frame indices, preprocessed pixels and prompt tokens, and the hosted
  checkpoint answered all 100 held-out videos exactly, with the offline evaluation's prompt-token count on each.
- `gemma-4-26b-a4b`: the trained `video_max_frames` and `video_frame_tokens`. Checkpoints trained at 8 x 70,
  32 x 70, 32 x 140 and 32 x 280 each served every held-out video with exactly the prompt tokens of its training rows.

For an evaluation set, check the files before the run starts (`datasets.upload_images`, inference.md).
