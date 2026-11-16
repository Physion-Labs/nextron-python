---
name: nextron-cloud
description: Run fine-tuning experiments on NexTron Cloud with the `nextron-cloud` Python SDK or CLI — choose a model, prepare and check JSONL training data, upload or import datasets from Hugging Face, estimate cost, start and watch training jobs, compare runs, download checkpoints and evaluate them. Use whenever the user wants to fine-tune, train, SFT or LoRA a model on NexTron, prepare data for it, check on a NexTron job, or run a training experiment. Also covers Serverless RL: running a user's own RL loop (this SDK's sample / train, or tinker / tinker-cookbook) on NexTron's GPUs.
---

# NexTron Cloud

NexTron Cloud trains open models on its own H100s, in two modes:

- **Managed training** (most of this guide): one request = one whole training run. You send a model, a dataset and a few
  settings; the service trains and leaves a checkpoint (a Hugging Face model directory) to download. Supervised
  fine-tuning (full or LoRA), reinforcement fine-tuning with built-in rewards (RFT, GRPO) and preference training (DPO). Billed by tokens trained.
- **Serverless RL** ([references/serverless-rl.md](references/serverless-rl.md)): an RL run is a loaded model the
  user's own loop samples from and trains step by step (`client.rl.connect`: `sample`, score in your code, `train`;
  or the `tinker` SDK and tinker-cookbook unchanged); NexTron runs each sample and each optimizer step. For custom
  RL environments, tools and rewards. Billed by time.

Nothing heavy runs on the user's machine in either mode. Text, and images for SFT on two models (references/data.md).

Everything below works from Python (`from nextron_cloud import NexTron`) or the shell (`nextron-cloud ...`, every
command prints JSON). Read the reference for each step when you reach it (`nextron-cloud guide <topic>` prints it,
e.g. `nextron-cloud guide data`; this page is `nextron-cloud guide`):

| step | read |
|---|---|
| choose a model, settings, cost and time; run a sweep | [references/training.md](references/training.md) |
| build, convert and check data; what is trained on | [references/data.md](references/data.md) |
| watch a job, read its metrics, fix a failure | [references/monitoring.md](references/monitoring.md) |
| download, serve and evaluate the checkpoint | [references/checkpoints.md](references/checkpoints.md) |
| reinforcement fine-tuning (RFT, a GRPO job): prompts with answers, built-in rewards | [references/rl.md](references/rl.md) |
| preference training (a DPO job): prompts with a chosen and a rejected reply | [references/dpo.md](references/dpo.md) |
| Serverless RL: your own RL loop (this SDK's `sample` / `train`, or the tinker SDK) on NexTron's GPUs, billed by time | [references/serverless-rl.md](references/serverless-rl.md) |
| try a trained model right after training, without downloading it: an inference run, called with this SDK or an OpenAI client | [references/inference.md](references/inference.md) |
| every SDK call and CLI command | [references/api.md](references/api.md) |

## Setup

```bash
uv venv && uv pip install --prerelease=allow nextron-cloud   # in a uv project: uv add --prerelease=allow nextron-cloud
export NEXTRON_API_KEY="nk_live_..."
uv run nextron-cloud me              # works = ready; prints the organisation and its balance in US cents
```

Use uv, not plain pip. Commands in this guide are written `nextron-cloud ...`: run them as `uv run nextron-cloud ...`
(Python as `uv run python ...`), or activate `.venv` first.

No key yet: accounts are by invitation (ask Physion Labs, support@physionlabs.ai). The owner signs in at
https://nextron.physionlabs.ai, opens **API keys** (only an organisation owner can create one) and adds credit under
**Billing**. Never print the key or write it into files you commit.

## Rules

1. **Jobs cost the owner money. Ask before starting one.** Run `jobs.estimate` with the exact arguments you will
   create with, tell the owner the cost (`credits` / 100 = US dollars), the model and the plan, and whether the data
   fits (the model's `max_seq_len`, the data's longest sample, `truncated_samples`), and wait for a yes.
   The CLI enforces this: `nextron-cloud jobs create ...` without `--yes` prints the estimate and starts nothing.
2. **Take models and limits from `client.models()`**, never from memory or this file: model ids, which methods each
   trains (`full`, `lora`, `grpo` for reinforcement learning, `dpo` for preference pairs), the longest sample per method, prices, GPU setups and whether one is free now.
   `jobs.estimate` runs the same checks as creating the job (it raises with the reason when the job would be refused)
   and returns the settings, about how many steps, and warnings: show those to the owner with the price.
3. **Never start the same job twice.** `jobs.create` retries safely with one idempotency key. If your own process
   died mid-create, pass the same `idempotency_key=` (CLI `--idempotency-key`) or check `jobs.list()` first.
4. **Check data before uploading** (`check_file`, CLI `datasets check`) and read what the server reports after its
   check (`dataset.errors`, `dataset.tokens`).
5. **Label every job**: `name=` says what the run tests, `tags=` groups runs (a sweep tags its jobs `sweep:<name>`).
   `jobs.list()` returns the whole organisation's jobs, other members' included; `jobs.list(tag=...)` finds yours.
   `nextron-cloud report --tag ...` compares them.
6. **Training and the checkpoint are separate steps.** `jobs.wait` returns when training is over; read the results
   then. The checkpoint uploads afterwards (minutes to an hour by model size) and stays downloadable for 30 days:
   `jobs.wait_checkpoint` / `download_checkpoint` only when the owner needs the weights.
7. **Poll slowly** (jobs every 30 s or more) and do not leave a job unwatched without telling the owner its id.
8. Images (JPEG, PNG) train on Qwen3.8-27B and Gemma-4-26B-A4B with SFT, for organisations they are opened to
   (references/data.md); audio and video are refused.

## An experiment, end to end

```python
from nextron_cloud import NexTron
from nextron_cloud.data import prepare_file

client = NexTron()

# 1. Model: pick one that trains the method and length the data needs, and is free or has a short queue.
for m in client.models():
    print(m.id, {k: (v.max_seq_len, v.price_per_mtok) for k, v in m.methods.items()},
          {l: (x.available, x.queued) for l, x in m.layouts.items()})

# 2. Data: the owner's own, converted, checked, split and uploaded; or a sample dataset (references/data.md).
prepare_file("tickets.csv", "data", prompt_column="ticket", response_column="reply")   # data/train.jsonl, data/val.jsonl
train = client.datasets.wait(client.datasets.upload("data/train.jsonl").id)
val = client.datasets.wait(client.datasets.upload("data/val.jsonl").id)
assert train.state == val.state == "ready", (train.errors, val.errors)

# 3. Cost, then the owner's yes.
draft = dict(model="qwen3.8-27b", dataset_id=train.id,
             params={"train_type": "lora", "learning_rate": 1e-4, "num_train_epochs": 2, "val_dataset_id": val.id})
quote = client.jobs.estimate(**draft)                 # quote.credits / 100 dollars; quote.enough

# 4. Train, watch, read the results. client.jobs.progress(job.id) says at any time where it is and how long is left.
job = client.jobs.create(**draft, name="support lora lr=1e-4 2ep", tags=["support-v1"])
job = client.jobs.wait(job.id)                        # or: for line in client.jobs.logs(job.id, follow=True)
print(job.state, job.error, job.notices, client.jobs.metrics(job.id)[-1], client.jobs.billing(job.id).charged_credits)

# 5. Later, when the owner wants the weights (kept 30 days).
client.jobs.wait_checkpoint(job.id)
client.jobs.download_checkpoint(job.id, f"checkpoints/{job.id}")
```

The same from the shell:

```bash
nextron-cloud models
nextron-cloud datasets prepare tickets.csv --out-dir data --prompt-column ticket --response-column reply
nextron-cloud datasets upload data/train.jsonl            # waits for the server's check
nextron-cloud jobs create --model qwen3.8-27b --dataset ds_... --param train_type=lora --param learning_rate=1e-4 \
    --name "support lora" --tag support-v1
# -> prints the estimate (exit code 3); after the owner agrees, the same command with --yes
nextron-cloud jobs progress job_...                       # phase, queue, ETA, upload bytes
nextron-cloud jobs events job_... --follow                # state changes, steps, validation, checkpoint stages
nextron-cloud sweep --name lr --model qwen3.8-27b --dataset ds_... --param train_type=lora \
    --grid learning_rate=5e-5,1e-4,2e-4                    # estimates; --yes starts all of them
nextron-cloud report --tag sweep:lr --table
nextron-cloud jobs wait-checkpoint job_... && nextron-cloud jobs download job_... checkpoints/job_...
```

A first run on new data should be a small one: `max_steps: 3` on the smallest GPU setup proves the data and settings
work for a fraction of the cost. Datasets of a few thousand short samples are smaller than one training step; read
"Small datasets" in training.md before planning them.
