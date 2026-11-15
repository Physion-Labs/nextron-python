# Changelog

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
