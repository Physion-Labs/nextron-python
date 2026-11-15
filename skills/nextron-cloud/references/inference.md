# Inference

An inference run serves the model a training job saved, on the GPUs of the cluster that trained it: try a fine-tune
right after training, and download it (checkpoints.md) only when it is good. A job's model stays there for 7 days
after training. Billed by time, like an RL run: from ready to stopped, per GPU-hour (`inference start` without
`--yes` prints the price); loading the model and stopping are not billed. A run stops itself after `--max-hours`, or
after `--idle-timeout` seconds without a request (default 1800).

```bash
nextron-cloud inference start --job job_... --max-hours 1          # the price only (exit code 3)
nextron-cloud inference start --job job_... --max-hours 1 --yes    # prints the run: inf_...
nextron-cloud inference wait inf_...                               # until the model is loaded, about 8 minutes
nextron-cloud inference url inf_...                                # the base URL for an OpenAI client
nextron-cloud inference stop inf_... --wait                        # always, when you are done
```

## Call it

Many prompts at once (an evaluation set): `connect` sends a whole batch in one request, as in an RL run.

```python
from nextron_cloud import NexTron

client = NexTron()
with client.inference.connect("inf_...") as live:
    batch = live.sample(prompts, n=1, max_tokens=512, temperature=0)   # prompts: strings or chat message lists
    replies = [group[0].text for group in batch]
```

Code written for OpenAI's API works with the run's base URL and your NexTron key:

```python
from openai import OpenAI

llm = OpenAI(base_url=client.inference.openai_base_url("inf_..."), api_key=os.environ["NEXTRON_API_KEY"])
reply = llm.chat.completions.create(model="any", messages=[{"role": "user", "content": "..."}], max_tokens=512)
```

- `chat/completions`, `completions` and `models`; any `model` name means the run's model.
- Chat prompts render with thinking off, as jobs train (data.md). Ask for it with
  `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`.
- A model that reasons before it answers (`muse-glimmer-30b`) returns the answer as the content and the reasoning
  apart: `reasoning_content` in a chat completion, `reasoning` on a sample. Give `max_tokens` room for both.
- `stream=True` streams the answer as it is generated; a long answer should stream. Without it, one call must finish
  in about 90 seconds through the public API: keep `max_tokens` moderate, and send large batches with `connect`
  instead (it polls, so it has no such limit).

The models that can be served list `inference` in `client.models()[i].methods`.
