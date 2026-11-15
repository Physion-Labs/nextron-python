# Managed reinforcement learning (GRPO)

This page is about a managed GRPO job: NexTron runs the loop and a built-in reward scores the replies. To run your
own loop with your own reward code, environments or tools, use Serverless RL
([serverless-rl.md](serverless-rl.md), `nextron-cloud rl ...`).

A GRPO job trains on prompts, not on replies to copy. Each round it takes `batch_size` prompts, the model writes
`num_generations` replies to each, a reward scores every reply, and the model learns from the difference between better
and worse replies to the same prompt. Full-parameter only. The models that
train it list `grpo` in `client.models()[i].methods`; use the job's `train_type="grpo"`.

The reward is built in (`boxed`, `exact`: 1 or 0 against the prompt's answer), or your own code (`remote`): NexTron
Cloud generates and trains, and each round's replies come to a script you run, which sends back their scores. Your code
runs on your machine, never on NexTron Cloud.

## Data

One kind per dataset: prompts with answers.

```json
{"prompt": "What is 17 * 23?\n\nPut your final answer within \\boxed{}.", "answer": "391"}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "answer": "391"}
```

- `answer` is a non-empty string (a number is kept as its text). With `messages`, the last message is the user's.
- From columns: `nextron-cloud datasets prepare data.csv --prompt-column question --answer-column final`.
- `check_file` reports `kind="rl"`; after upload `dataset.kind == "rl"` and `dataset.tokens[model]["max"]` is the
  longest prompt. It must be at most the method's `max_prompt_len`.
- A file mixing chat samples and RL samples is refused. A `{"question", "answer"}` row is a chat sample (the answer is
  the reply to learn); add `prompt` or `messages` to make it RL.

## Rewards

| `reward` | a reply scores 1 when |
|---|---|
| `boxed` | the last `\boxed{...}` in it equals the answer: numbers compared as numbers (`1,000` = `1000`), a single option letter ignoring case, other text ignoring case and `$`. Ask for `\boxed{}` in the prompt. |
| `exact` | the whole reply, trimmed, equals the answer. Ask for the answer alone in the prompt. |
| `remote` | your function says so: any score, from tests, a judge model or anything else (below). |

The model answers in non-thinking mode (no `<think>` block), as in SFT.

## Your own reward (`remote`)

Not on the service yet: a model lists the rewards it takes in `client.models()[i].methods["grpo"].rewards`. Until
`remote` is there, your own reward code runs with Serverless RL ([serverless-rl.md](serverless-rl.md)).

Start the job with `reward="remote"`, then run your scorer until the job ends:

```python
def reward(item):                      # one reply; return a float (higher is better)
    tests = item["metadata"]["tests"]  # item: messages (the prompt), completion, row, answer, metadata
    return passed(item["completion"], tests) / len(tests)

client.jobs.serve_rewards(job.id, reward)      # blocks; returns when training is over
```

or `nextron-cloud jobs serve-rewards job_... reward.py:reward`. Put what the scorer needs in each sample's `answer` and
optional `metadata` (any JSON: test cases, a rubric); `row` is the sample's position in the dataset.

- Each round waits for all its scores. A round without them for 10 minutes fails the job, and no checkpoint is saved:
  start the scorer right after creating the job and keep it running. The time your scorer takes adds to every
  round; score in parallel (threads, a pool) when it is slow.
- Replies arrive in batches, one per GPU per round (`jobs.rollouts(id)` lists those waiting, `jobs.score(id, batch,
  scores)` answers one: what `serve_rewards` does). Scores must be finite, one per reply, in order.
- Run untrusted model output (generated code) in a sandbox on your side.
- Scores of any scale work: GRPO compares the replies to the same prompt. The `reward_mean` metric is the mean of your
  scores.

## Params

| key | default | allowed / notes |
|---|---|---|
| `reward` | required | `"boxed"`, `"exact"`, `"remote"` |
| `max_steps` | required | rounds; one optimizer step each |
| `batch_size` | 64 | prompts a round, a multiple of 8, at most 256 |
| `num_generations` | 8 | replies per prompt, 2-16 |
| `max_completion_length` | 8192 | tokens a reply may have, at most the method's `max_completion_length`. A reply cut here usually scores 0: Qwen3.8-27B's math replies average ~3,500 tokens, and a short DAPO-Math run at 4096 scored ~0.3 against ~0.8 at 8192 |
| `temperature` | 1.0 | sampling temperature of the replies |
| `learning_rate` | 1e-6 | |
| `seed` | 42 | |

`num_train_epochs`, `warmup_ratio`, `weight_decay`, `lr_scheduler_type`, `max_seq_len`, `val_dataset_id` and
`eval_steps` do not apply and are refused.

```bash
nextron-cloud jobs create --model qwen3.8-27b --dataset ds_... --param train_type=grpo --param reward=boxed \
    --param max_steps=3 --name "math grpo smoke" --tag math-rl     # prints the estimate; --yes to start
```

## Plan the first run

GRPO learns only from prompts the model sometimes gets right: when all replies to a prompt score the same, that prompt
teaches nothing. Start with a short run (`max_steps` 2-3) and read its metrics before a long one:

- `reward_mean`: the share of replies scored 1. Between about 0.1 and 0.9 the model has room to learn.
- `zero_variance_fraction`: the share of prompts whose replies all scored the same. Near 1, nothing is learned: pick
  harder (or easier) prompts.
- `completion_length_mean` close to `max_completion_length`: many replies are cut before their answer. Raise the limit.

For example, Qwen3.8-27B on DAPO-Math (competition math) starts at reward_mean ~0.81 with ~0.7 of prompts giving no
signal: a run there learns from the remaining ~30%.

## Cost and time

Billed per token at the method's price: every reply and its prompt, all rounds. The estimate holds the most a run could
use, `max_steps x batch_size x num_generations x (longest prompt + max_completion_length)` tokens, and the charge is what
was generated, usually much less (replies are shorter than the limit). `jobs.billing(id)` shows both.

Measured on Qwen3.8-27B, one 8x H100 node, 64 prompts x 8 replies, replies averaging 3,400 tokens: about 5 minutes a
round, ~60% of it generating and ~30% training. `jobs.progress` counts whole rounds in its time left.

## Watch

Each point of `jobs.metrics(id)` is one round: `reward_mean`, `zero_variance_fraction`, `completion_length_mean`,
`generation_s` and `round_time_s`, besides the step's `loss`. `jobs.events` has a
`training.rollout` event per round.

Trouble looks like: reward_mean falling and staying down over several rounds, replies growing to the limit or
shrinking to almost nothing. Rounds draw different prompts, so one round's reward moves by
several points on its own; judge the trend. Cancel with `jobs.cancel` (charged for the tokens so far).

Only the final model is saved, and a cancelled job leaves none: keep runs short and run again with more `max_steps`
once a short one looks right. There is no validation reward during training; measure the checkpoint on held-out prompts
afterwards (checkpoints.md).
