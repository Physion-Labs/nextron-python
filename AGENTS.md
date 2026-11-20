# Agent guide

**To run fine-tuning experiments on NexCloud** (no other context needed): read
[skills/nextron-cloud/SKILL.md](skills/nextron-cloud/SKILL.md) first, then the reference it points to for each step.
It covers setup and the API key, the rules (ask the owner before anything that costs money), data, training,
monitoring, checkpoints and evaluation. Install with uv: `uv venv && uv pip install --prerelease=allow nextron-cloud` (or `uv add --prerelease=allow
nextron-cloud` in a uv project; `uv sync` in a clone of this repository), then run commands as `uv run nextron-cloud ...`.

**Working on this repository:**

- Layout: `src/nextron_cloud/`: `_client.py` (HTTP, retries, errors, `me`/`models`/`layouts`), `datasets.py`,
  `jobs.py`, `rl.py` (Serverless RL: `client.rl`), `live.py` (an RL run's `sample` / `train` calls), `data.py` (sample format, local check, `prepare_file`), `experiments.py` (sweep, report), `types.py`
  (response models; unknown fields are kept), `cli.py` (`nextron-cloud`). `examples/`: scripts outside the package.
- Checks: `uv run pytest`, `uv run mypy`. Against a live deployment, with a key: `uv run python scripts/verify_live.py`
  (reads, an estimate, two dataset uploads that it deletes; starts no job).
- `data.py` must accept and reject exactly what the server's format check does. Change both together.
- The API is the contract: read only fields the server returns. Model lists, limits and prices come from the API at
  run time, never from constants here.
- A change to the API surface updates `skills/nextron-cloud/references/api.md` in the same commit.
- This repository is published. Write every commit, comment and doc for outside readers: no internal hosts, clusters,
  people or incidents.
