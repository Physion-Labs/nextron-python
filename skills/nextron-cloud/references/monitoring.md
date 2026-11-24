# Monitoring

## States

`queued` (waiting for GPUs, or for a large file's samples to be measured: data.md) → `submitting` → `pending` (the
cluster is starting it) → `running` → `succeeded` | `failed` | `cancelled`. `job.done` is true for the last three:
training is over. A succeeded job's checkpoint then uploads (`checkpoint.status` `uploading` → `ready`,
`job.checkpoint_ready`); that is a separate wait (`jobs.wait_checkpoint`), needed only to download the weights.

`jobs.get(id)` (not `list`) also has:

- `timeline`: unix times of `queued`, `starting` (the cluster started it), `training` (step 1 began), `trained` (last
  step ended), `saved` (checkpoint written), `done` (checkpoint uploaded, or the job ended). A missing value has not
  happened yet.
- `notices`: sentences worth reading, each with a `level` and a `code` (`SAMPLES_TRUNCATED`, `SAMPLES_SKIPPED`,
  `BATCH`, `BATCH_LARGER_THAN_ASKED`, `FEW_TOKENS_PER_GPU`, `GPU_OUT_OF_MEMORY`): samples truncated or skipped, the batch actually used, too
  few tokens per GPU, a GPU out of memory.
- `tokens_trained`: what the job is charged for.
- `metrics`: the latest step (`step`, `total_steps`, `loss`, `tokens_per_s_per_gpu`, ...).
- `checkpoint`: `{"status": "uploading" | "ready", "step", "files", "bytes"}` once it is being saved.
- `config`: the training config the engine ran (paths masked).

## Watching

```python
p = client.jobs.progress(job_id)          # where it is now, in one answer
for event in client.jobs.events(job_id, follow=True):               # what happens, in order, until training is over
    print(event.type, event.data)
job = client.jobs.wait(job_id, timeout=None, poll_interval=60)      # or just wait for succeeded, failed or cancelled
points = client.jobs.metrics(job_id)                                # one point per step so far
for line in client.jobs.logs(job_id, follow=True):                  # the raw log, for debugging
    print(line.line)
```

`progress` (CLI `jobs progress ID`):

- `phase`: `queued` → `starting` → `loading` (model load and kernel compile) → `training` → `saving` →
  `checkpoint_queued` (other jobs' uploads first) → `checkpoint_preparing` → `checkpoint_uploading` → `done`; or
  `failed` / `cancelled`.
- `jobs_ahead`: queued jobs that start before this one.
- `training`: `step`, `total_steps`, `fraction`, `seconds_per_step` (median, step 1 excluded), `eta_seconds`, `loss`.
- `checkpoint`: `stage`, `bytes_total`, `bytes_done`, `fraction`, `bytes_per_s`, `eta_seconds`, `uploads_ahead`.
- `last_activity` / `idle_seconds`: when anything last happened. Minutes of silence in `training` or
  `checkpoint_uploading` mean something is stuck: report the job id; a long `queued` or `checkpoint_queued` is waiting,
  not stuck.

`events` (CLI `jobs events ID [--after SEQ] [--follow] [--until checkpoint]`, JSON lines): `job.state` (`data.state`,
and `error` on failure), `job.cancel_requested`, `training.step` (`step`, `total_steps`, `loss`, `step_time_s`),
`training.validation` (`step`, `val_loss`), `checkpoint.stage` (`queued`, `converting`, `hashing`, `transferring`,
`ready`). Each has a `seq`: pass the last one as `after` to read on after a restart. `follow` stops when training is
over; `until="checkpoint"` also waits for the upload.

A metric point: `step`, `total_steps`, `loss` (training loss of that step), `lr`, `step_time_s`,
`tokens_per_s_per_gpu`, `tokens_per_s_total` (real tokens, all GPUs), `ts`, `val_loss` on steps where the validation
set was run, and the step's `samples`, `real_tokens` (every token of its samples; what is billed), `trained_tokens`
(the assistant tokens the loss is computed on) and `padding_tokens`.

`job.metrics` holds the latest step only (same throughput as `jobs.metrics()`). Metrics, progress and events follow the
log by up to a minute.

Notices: "Batch: about N samples per step (8 GPUs, each packing 64K tokens)" gives the samples a step takes and each
GPU's row capacity, not the tokens actually in it. Log lines about the validation set's own batch refer to the
validation pass, not to training.

What to check:

- Step 1 takes much longer than the rest (start-up compilation); judge speed from step 2 on.
- Step 1 loss finite and not 0. Its size depends on the data: near 0 for answers the base model already gives
  (short, formulaic), around 1-3 for new content. NaN means a data or settings problem.
- Loss going down over the first tens of steps. A flat or rising loss: learning rate too high (rising, spiky) or too
  low (flat).
- `val_loss` turning up while `loss` keeps falling: overfitting; fewer epochs or a lower learning rate.
- Throughput near the model's `tokens_per_s_per_gpu` from `models()`; much lower with a small `batch_size` or a
  dataset smaller than a step is expected. Read throughput from `jobs.metrics()`, which counts every GPU exactly;
  `job.metrics` (the job's latest step) is a quicker, rougher figure.

`jobs.export(job_id, "job.zip")` saves the whole record (job, config, logs, metrics, GPU telemetry, manifest). The
manifest exists only once the checkpoint is uploaded (`jobs.manifest` answers 409 before that), so export after
`wait_checkpoint` for the complete record.
`jobs.telemetry(job_id)` returns GPU and host utilisation while it trained.

## Cancelling

`nextron-cloud report JOB_ID ...` (or `--tag`) puts several jobs side by side.

`jobs.cancel(job_id)`: a queued job is cancelled at once; a running one stops soon after. A cancelled job
leaves **no checkpoint** and pays for the tokens trained before it stopped. There are no intermediate checkpoints:
only a job that finishes all its steps has one.

## When a job fails

`job.error` is short ("training job failed; see the logs"). Then:

1. `job.notices`: an `error` notice names the cause when it is known (`GPU_OUT_OF_MEMORY`, with the fix).
2. The last 50 log lines: `list(client.jobs.logs(job_id))[-50:]`. Look for `Error`, `Traceback`, `out of memory`.
3. Out of memory: a smaller `batch_size` or `max_seq_len` (or the model's long-context version for long samples).
4. Anything else, or a failure that repeats with the same settings: report it to Physion Labs with the job id. A
   failed job is not charged; retrying it is the owner's call.

## Errors from the API

`APIError` (CLI: `{"error", "status", "message"}` and exit code 1):

| status | meaning | what to do |
|---|---|---|
| 401 | key missing or wrong | check `NEXTRON_API_KEY` |
| 402 `insufficient_balance` | the balance does not cover the estimate | the owner adds credit (website, Billing) |
| 403 | the key cannot do this (keys and billing are website-only) | |
| 404 | no such job or dataset in this account | check the id |
| 409 | not ready (dataset still checking, checkpoint still uploading, a dataset in use) | wait, retry later |
| 410 | checkpoint expired (kept 30 days) | |
| 422 | a setting or the data is wrong; `message` says which | fix that setting |
| 429 | 10 open jobs, 50 datasets, or over 300 requests a minute | wait, or delete old datasets |
| 5xx | service trouble; the SDK already retried | retry later |
