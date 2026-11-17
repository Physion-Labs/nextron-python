# Serverless RL

NexTron Cloud has two modes.

| | Managed training | Serverless RL |
|---|---|---|
| You send | a dataset and settings | prompts and rewards from your own loop |
| NexTron runs | the whole training run | every sample and every optimizer step, on the run's GPUs |
| Client | this SDK: `jobs` | this SDK: `rl.connect`; or the `tinker` SDK, pointed at the run |
| Billed by | tokens trained | time, from ready to stopped |
| Use it for | fine-tuning on examples; RL with a built-in reward | your own RL loop: environments, tools, multi-turn episodes, your reward code |

An RL run is a model loaded on a node of GPUs, kept for you until you stop it. Your loop, environments and reward code run on your machine; NexTron
samples with vLLM and trains all of the model's weights (not an adapter) on the same GPUs. There are two ways to drive
one: NexTron's own calls (below, the default) and Tinker's API, for code already written for the `tinker` SDK.

A model that has Serverless RL lists `interactive` under its methods in `nextron-cloud models`, with
`price_per_gpu_hour`; `rl start` answers "coming soon" for one without it.

## Start and stop

```bash
nextron-cloud rl start --model qwen3.8-27b --max-hours 2                   # the price only: exits 3, starts nothing
nextron-cloud rl start --model qwen3.8-27b --max-hours 2 --yes             # after the owner agreed
nextron-cloud rl wait rl_...                                               # 6-8 minutes (about 20 for qwen3.8-next-180b); progress on stderr
...                                                                        # your loop, one of the two ways below
nextron-cloud rl stop rl_... --wait                                        # stop billing; returns when the GPUs are released
```

An RL run is priced per GPU-hour, by the kind of GPU: `rl start` without `--yes` prints it, e.g. $3.50 per H100-hour
for a run of 8 GPUs. Numbers are US cents: `price_per_gpu_hour` 350, `gpus` 8, and `credits` is what is held for
`--max-hours` (350 x 8 x hours). You pay for `billable_seconds` on all of the run's GPUs; the rest of the hold is
released.

`qwen3.8-next-180b` runs on two nodes (16 GPUs, so twice the price an hour). Its model takes about 20 minutes to load,
and when the run stops it saves its checkpoint (360 GB to download) for about 15 minutes in `closing` (not billed).

## Your loop with this SDK: sample, score, train

```python
from nextron_cloud import NexTron

client = NexTron()
run = client.rl.wait(client.rl.start(model="qwen3.8-27b", max_hours=2).id)
try:
    with client.rl.connect(run.id) as live:
        for step in range(20):
            batch = live.sample([p["question"] for p in problems], n=8, max_tokens=512)   # one call, every prompt
            rewards = [[score(p, reply.text) for reply in group] for p, group in zip(problems, batch)]   # your code
            result = live.train(batch, rewards, learning_rate=2e-6)                        # one optimizer step
            print(step, result.reward_mean, result.zero_variance_fraction, result.client_s)
        live.finish()                       # saves the model and stops the run
finally:
    client.rl.stop(run.id)                  # also when your loop fails
```

- `sample`: a prompt is a string (one user message), a list of chat messages, or `{"messages": [...], "tools": [...]}`;
  the run renders it with the model's chat template, so you need no tokenizer. All prompts of a call are generated
  together. `batch[i][j]` is reply `j` to prompt `i`: `.text`, `.finish_reason` (`stop` or `length`), `.tokens`.
- `train`: one reward per reply. The run kept each reply's tokens and sampler logprobs, so only the rewards
  travel. The advantage of a reply is its reward minus the mean reward of the replies to the same prompt
  (`advantage="centered"`); `"normalized"` also divides by their standard deviation, `"raw"` uses the reward as it is.
- Several turns (tools, an environment that answers): sample each turn with the conversation so far as the prompt,
  then train on episodes: `live.train(episodes=[Episode([turn1, turn2], reward=1.0, group="task-7")], ...)`. An
  episode has one reward; `group` names the episodes it is compared with.
- Per call: `learning_rate` (required), `loss` (`importance_sampling`, `ppo`, `cispo`; `clip_low` / `clip_high`),
  `substeps` (optimizer steps the episodes are split over), `beta1`, `beta2`, `eps`, `weight_decay`.
- The weights change with every `train`: replies sampled before it cannot be trained after it.
- `result` is the step's record, and the run keeps one per training call (`client.rl.metrics(run_id)`):
  - how it went: `reward_mean`, `reward_std`, `zero_variance_fraction` (groups whose replies all got the same reward
    teach nothing: mostly `all_best_fraction` means the task is too easy, `all_worst_fraction` too hard),
    `truncated_fraction` (replies cut by `max_tokens`), `completion_length_mean`, `kl_sample_train` (near 0 when the
    sampler and the trainer agree; a rising value says the learning rate is too high);
  - where the time went: `idle_s` before the step, then `step_s` = `generation_s` + `train_s` + `client_s`. `client_s` is the run waiting for
    you (scoring, the network), and it is billed like the rest: speed up your reward code if it is large;
  - what was trained: `tokens`, `tokens_per_s_per_gpu`, `rows`, `row_fill`; and `session_s`, about what is billed so far.
- Look at what the model writes: `nextron-cloud rl rollouts rl_...` (`client.rl.rollouts(id, step=N)`)
  shows the best and the worst group of a training call with their rewards, kept by the run itself.
- Keep a record of the run: `client.rl.connect(id, log_dir="runs/math-1")` appends every step's record to
  `metrics.jsonl` and the first groups it trained (prompts, replies, rewards) to `rollouts.jsonl`. Read a few
  rollouts every so often: a reward that rises while the replies get worse is the model gaming your reward code. For
  Weights & Biases, `wandb.log(result.model_dump())` after each `train`.
- Size your steps to fill the trainer's rows. A step's sequences are packed into rows of at most `result.row_tokens`
  tokens on each GPU, and a row costs a few seconds however full it is. `result.row_fill` says how full they were:
  aim for 0.9 or more by sampling `gpus * rows * row_tokens` tokens a step (prompts included) for a whole number of
  `rows`. Just over a multiple is the worst case: one more row on every GPU, each partly empty. `row_tokens` differs by
  model, and a mixture-of-experts model (`gemma-4-26b-a4b`) pads every row to a fixed length: there a partly filled
  row costs as much as a full one, and the first step at a row length takes 20-50 s longer than the ones after it.
- An open `LiveRun` tells the run you are alive every 20 seconds, so slow reward code is fine; if your process dies
  the run stops 3 minutes later.

## With the tinker SDK

An RL run also serves Tinker's HTTP API: code written for the `tinker` SDK (your own, or
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)'s loops and recipes) runs unchanged once
two environment variables point at the run. It sends one request per prompt and the sampled tokens back with
every step, so a step takes longer on the wire than with the calls above.

```bash
eval "$(nextron-cloud rl env rl_...)"                                      # TINKER_BASE_URL, TINKER_API_KEY
pip install tinker tinker-cookbook
python -m tinker_cookbook.recipes.math_rl.train env=gsm8k model_name=Qwen/Qwen3.8-27B \
    renderer_name=qwen3_8_disable_thinking loss_fn=importance_sampling learning_rate=1e-6 \
    group_size=4 groups_per_batch=16 max_tokens=1024 max_steps=3 eval_every=0 save_every=0
```

In Python: `os.environ.update(client.rl.env(run.id))` before `import tinker` (it holds your key: do not
print it). [nextron-cookbook](https://github.com/Physion-Labs/nextron-cookbook) has tested settings for cookbook
recipes (`nextron-cookbook run math`), measured results and the full compatibility table.

## Rules

- An RL run costs money from the moment it is ready until it is stopped, whether or not anything trains. Show the
  owner the price first (`rl start` without `--yes`), and always stop it: when your loop ends, and when it fails.
- It also stops itself: when your training client ends its session (with the `tinker` SDK: the end of a
  `with tinker.ServiceClient() as service:` block, or `service.close("success").result()`; the state becomes
  `closing` within about 15 seconds), after `--max-hours`, and after `--idle-timeout` seconds (default 1800) without
  a sample or a training step. A client that goes away without ending its session leaves the run ready, and billed,
  until 3 minutes after its last heartbeat: a killed process, and also a script that made a bare
  `tinker.ServiceClient()` and returned (the `tinker` SDK does not end the session when the process exits). Stop it
  yourself to stop billing sooner, and in scripts stop it in a `trap` or `finally`.
- Billing stops when the run begins to stop (state `closing`): your `rl stop`, the client finishing, or the limits
  above. The engine then saves the model if it trained and releases the GPUs about a minute later (`closed`). A new
  run of the same model may wait in `queued` until then.
- One training loop per RL run, driven one way: a second training client is refused, and so is the `tinker` SDK on a
  run used through `rl.connect` (and the other way round). Start another RL run for another loop.
- `rl env` never prints your key (its line refers to `$NEXTRON_API_KEY`). `client.rl.env()` returns it for
  `os.environ`: never print or log that value.
- Learning rates are for full-parameter training: 1e-6 to 2e-6 for RL on a 27B model, about 10x below LoRA defaults.

## Settings

| `rl start` | Default | |
|---|---|---|
| `--model` | | the model to load |
| `--max-hours` | required | stops itself after this long; credits for it are held at the start |
| `--max-context` | 16384 | tokens: a prompt with its reply, and a training sequence. Smaller leaves more memory for parallel sampling |
| `--max-grad-norm` | 1.0 | gradient clipping for every optimizer step |
| `--no-grad-clip` | clipping on | no gradient clipping (`grad_clip=False`). The optimizer then runs beside the backward pass and a step is about 2 s shorter; clipping is a guard against the rare very large gradient, so keep it on unless your runs are stable |
| `--idle-timeout` | 1800 | seconds without a sample or a step before it stops |

Per step, from your loop: Adam's `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay` and the loss
(`importance_sampling`, `ppo`, `cispo` with their clip thresholds; through Tinker's API also `cross_entropy`).

## States and the trained model

`queued` -> `starting` (the model loads, 6-8 minutes, about 20 for qwen3.8-next-180b; not billed) -> `ready` -> `closing` (not billed) -> `closed`, or
`failed`. `rl get` shows `billable_seconds`. A failed run is not charged.

Two kinds of id appear: `rl_...` is the RL run, for every `nextron-cloud` command. The `tinker` SDK prints its own
ids and paths (`nx-...`, `tinker://nx-.../weights/final`); they are not arguments to anything here.

When an RL run that trained at least one step stops, the model is saved and uploaded like a job's checkpoint. The
run's id works with the job commands: `nextron-cloud jobs wait-checkpoint rl_...`, `jobs download rl_...`,
`jobs logs rl_...`.

## What is not there yet

Either way: sampling from earlier weights, named checkpoints during a run, LoRA, images, a second model in the same
run, per-token advantages. Through `rl.connect`: supervised steps (`cross_entropy`) and scoring text without
training; Tinker's API has both. Through Tinker's API: `loss_fn=stable_reinforce` (pass `importance_sampling` to
cookbook recipes whose default it is), `kl_penalty_coef`, custom losses. Unsupported calls fail with a message that
says so.
