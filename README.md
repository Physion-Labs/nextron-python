# nextron-cloud

[![ci](https://github.com/Physion-Labs/nextron-python/actions/workflows/ci.yml/badge.svg)](https://github.com/Physion-Labs/nextron-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nextron-cloud?include_prereleases)](https://pypi.org/project/nextron-cloud/)
[![Python](https://img.shields.io/pypi/pyversions/nextron-cloud)](https://pypi.org/project/nextron-cloud/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

> **Release candidate.** The API may still change before 0.1.0; install with `--prerelease=allow`.

Python SDK and CLI for [NexCloud](https://nextron.physionlabs.ai), which trains open models in two modes and
serves what it trained:

- **Managed training**: one request (model + dataset + settings) fine-tunes a model and leaves a checkpoint to download:
  supervised fine-tuning (full parameter or LoRA), preference training (DPO) and reinforcement fine-tuning with a
  built-in reward (RFT). This page starts there.
- **Serverless RL**: a loaded model your own RL loop samples from and trains, step by step:
  `client.rl.connect(id)` with `sample` and `train`, or the `tinker` SDK and tinker-cookbook unchanged; see
  [serverless-rl.md](skills/nextron-cloud/references/serverless-rl.md).
- **Inference**: a finished job's model served on the GPUs that trained it, before you download it, called with this
  SDK or any OpenAI client; see [inference.md](skills/nextron-cloud/references/inference.md).

```bash
uv venv && uv pip install --prerelease=allow nextron-cloud   # or, in a uv project: uv add --prerelease=allow nextron-cloud
export NEXTRON_API_KEY="nk_live_..."                           # API keys page on nextron.physionlabs.ai
uv run nextron-cloud models                                    # what can be trained now, at what price
```

Commands below are written as `nextron-cloud ...`; run them with `uv run` (or activate `.venv`).

```python
from nextron_cloud import NexTron

client = NexTron()
dataset = client.datasets.wait(client.datasets.upload("train.jsonl").id)   # checked locally, then by the server
# no data yet? a sample dataset: next(d for d in client.datasets.list() if d.sample and not d.validation)
draft = dict(model="qwen3.8-27b", dataset_id=dataset.id, params={"train_type": "lora", "learning_rate": 1e-4})
print(client.jobs.estimate(**draft))                                        # cost in US cents: get a yes first

job = client.jobs.wait(client.jobs.create(**draft, name="first try").id)  # only after the cost is approved
print(job.state, client.jobs.metrics(job.id)[-1])

client.jobs.wait_checkpoint(job.id)                                         # later: weights stay 30 days
client.jobs.download_checkpoint(job.id, f"checkpoints/{job.id}")
```

## Start with an agent

Install, set the key, then paste this to your coding agent (Claude Code, Codex, ...). It asks what you need, then
prepares the data and shows you the cost before it starts anything.

```text
Run a training experiment for me on NexCloud, a managed training service: one request fine-tunes a model on
its GPUs (on example replies, on pairs of a preferred and a rejected reply, or with reinforcement learning against
known answers) and leaves a checkpoint to download.

Setup
- Install the SDK with uv in this project: `uv venv && uv pip install --prerelease=allow nextron-cloud`
  (in a uv project: `uv add --prerelease=allow nextron-cloud`). Run its commands with `uv run`, e.g.
  `uv run nextron-cloud guide`.
- Source and docs: https://github.com/Physion-Labs/nextron-python
- My API key is in the environment variable NEXTRON_API_KEY. Never print it or write it into a file.
- Before anything else run `uv run nextron-cloud guide` and read all of it. At each step read its topic too:
  `nextron-cloud guide data`, `... training`, `... rl`, `... dpo`, `... monitoring`, `... checkpoints`,
  `... inference`, `... api`. Follow its rules.

The experiment
- Model: ask me, or pick one from `nextron-cloud models` and tell me why.
- After reading the guide, ask me what I need before you prepare anything, a few questions at a time: what the
  model should do better; whether to train on my own data (a file or a Hugging Face link, and which fields are the
  input and which the answer) or on one of the sample datasets (`nextron-cloud datasets list --sample --table`);
  how I will judge the result; and my budget. Suggest answers where you can, e.g. a held-out split of my data
  compared with the base model. If a reply can be checked against a known answer (math, multiple choice, short
  exact answers), say whether reinforcement learning (`nextron-cloud guide rl`) fits better than example replies.
- Spend at most the budget I give you unless I approve more.

How to work
- Prepare and check the data, hold out a validation set, upload, and plan a small first run (for reinforcement
  learning, 2-3 rounds to read the starting reward).
- Before anything that costs money, show me the model, settings, estimated cost, steps, time and warnings, and
  how long the data is: its tokens and longest sample, the model's max_seq_len and how many samples would be
  truncated. Wait for my yes. Never start a job I have not approved.
- Give every job a name and a tag for this experiment.
- While jobs run, watch them (`nextron-cloud jobs progress` / `jobs events`) and tell me at once if one fails or
  stalls.
- When training ends, report loss and validation loss (reinforcement learning: the reward per round; preference
  pairs: the loss, and the answers on held-out prompts), what was charged, what you conclude, and what you would try next. Download checkpoints only when I ask.
- Keep everything you write in this project directory.
```

## Documentation

The guide is written for people and coding agents alike, in [skills/nextron-cloud](skills/nextron-cloud/SKILL.md):

- [SKILL.md](skills/nextron-cloud/SKILL.md): setup, rules, an experiment end to end
- [data.md](skills/nextron-cloud/references/data.md): the sample format, what training learns from, checks
- [media.md](skills/nextron-cloud/references/media.md): images and videos: models, format, video settings, cost
- [training.md](skills/nextron-cloud/references/training.md): models, params, batch, cost and time, sweeps
- [monitoring.md](skills/nextron-cloud/references/monitoring.md): states, metrics, failures, API errors
- [rl.md](skills/nextron-cloud/references/rl.md): reinforcement fine-tuning (RFT, a GRPO job) with built-in rewards
- [dpo.md](skills/nextron-cloud/references/dpo.md): preference training (a DPO job) on pairs of a chosen and a rejected reply
- [checkpoints.md](skills/nextron-cloud/references/checkpoints.md): download, serve, evaluate
- [serverless-rl.md](skills/nextron-cloud/references/serverless-rl.md): Serverless RL: your own RL loop with `sample`
  and `train`, or the `tinker` SDK and tinker-cookbook, on NexCloud's GPUs
- [inference.md](skills/nextron-cloud/references/inference.md): an inference run: try a trained model without downloading it
- [api.md](skills/nextron-cloud/references/api.md): every SDK call and CLI command
- [examples/compare_outputs.py](examples/compare_outputs.py): answer held-out prompts with the fine-tuned and the base
  model side by side

Claude Code: `/plugin install nextron-cloud --marketplace Physion-Labs/nextron-python` installs it as a skill.
Other agents: point them at [AGENTS.md](AGENTS.md).
