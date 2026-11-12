# Changelog

## 0.1.0rc1

First release candidate.

- `NexTron` client: `me`, `models`, `layouts`; datasets (upload with a local format check, Hugging Face import,
  wait, download link, delete); jobs (estimate with the server's checks, create with name and tags, wait, progress,
  events, logs, metrics, billing, cancel, checkpoint download with sha256 checks, export).
- Data helpers: `to_messages`, `write_jsonl`, `check_file`, `prepare_file` for messages, ShareGPT, Alpaca, Dolly,
  OpenOrca, prompt/completion, question/answer, problem/solution and input/output rows.
- Experiments: `sweep` (one job per combination, repeat-safe) and `report`.
- `nextron-cloud` CLI, JSON output; `nextron-cloud guide` prints the agent guide shipped in the package.
