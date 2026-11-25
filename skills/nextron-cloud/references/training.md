# Training

## Choose a model

`client.models()` (CLI `nextron-cloud models`) is the live list. Per model:

- `methods`: the ways it trains or serves (`full`, `lora`, `dpo`, `grpo`, `interactive` for Serverless RL, `inference`). `full` and `lora` each have `max_seq_len` (the longest sample it trains), `price_per_mtok` (US cents
  per million tokens trained) and `layouts` (GPU setups, in the order `layout="auto"` tries them).
- `layouts[id]`: `gpus`, `tokens_per_s_per_gpu`, `available` (a job would start now), `queued` (jobs ahead).
- `batch_size` / `validation_set`: whether `params.batch_size` / `params.val_dataset_id` are accepted. Large
  expert-parallel models set their own batch and take no validation set.
- `methods[m].defaults`: the optimizer and step settings a job trains with unless it sets them (`max_grad_norm`, 0 = no
  clipping; `adam_beta1`, `adam_beta2`, `adam_epsilon`; for LoRA `lora_rank`, `lora_alpha`, `lora_dropout`), and
  `methods[m].max_tokens_per_gpu`: the largest `tokens_per_gpu` (null where the model trains with its tuned step).
- `name`, `maker`, `params`, `active_params`, `architecture`.

Pick by: the method you need, a `max_seq_len` that holds the data (`dataset.tokens[model].max`), price, and queue.
Longer-context versions are separate ids (for example `...-256k`) with their own price.

## Methods

- **full**: every weight trained; FP32 master weights, gradients and AdamW states, BF16 compute.
- **lora**: rank 32, alpha 64, on every linear layer of the language model (routed MoE experts excluded); base
  weights frozen. The adapters are **merged** into the checkpoint, so it is a full model like a full fine-tune.
  Most models have it; `kimi-k3.1-pro`, the Olmo-3 models and the long-context ids (`...-256k`, `...-1m`) do not
  (a model's `methods` say so). The larger models train LoRA on fewer GPUs than a full fine-tune: with `layout="auto"` the job takes the
  smallest setup that trains it.

### Start from a job's model

A job can train on from the model another job saved instead of from the base model: `from_job`, on every method
(`--param from_job=job_...`) and on an RL run (`rl start --from-job job_...`). Chain them as you like: SFT then RFT,
LoRA on a full fine-tune, a second pass on new data.

```bash
nextron-cloud jobs create --model qwen3.8-27b --dataset ds_... --method lora --param from_job=job_... --yes
```

The job must be yours, a training job of the same model that succeeded, and its checkpoint must still be stored (30
days after training). In the first 7 days the model is still on the cluster that trained it and the new job starts at
once; after that it is fetched from storage first, which takes some minutes and is not billed. The models with expert
parallelism (120B and up) do not take it yet: `nextron-cloud models` shows `from_job` for each model. The price is the
model's own.

## Params

All optional; the server fills the rest from a tested recipe for the model.

| key | default | allowed / notes |
|---|---|---|
| `train_type` | `"full"` | `"full"`, `"lora"`; `"grpo"` and `"dpo"` have their own settings ([rl.md](rl.md), [dpo.md](dpo.md)) |
| `num_train_epochs` | 1 | 1-20. Give this or `max_steps` |
| `max_steps` | | 1-100,000. With both, training stops at whichever comes first (`num_train_epochs` is 1 unless set) |
| `learning_rate` | 1e-5 | up to 1e-3. Starting points: full 1e-5, LoRA 1e-4 |
| `lr_scheduler_type` | `"cosine"` | `"cosine"`, `"linear"`, `"constant"` |
| `warmup_ratio` | 0.03 | 0-0.5 |
| `weight_decay` | 0.1 | 0-1 |
| `batch_size` | see below | samples per optimizer step, multiple of 8 and of the GPU count; only where `models()` says `batch_size` |
| `max_seq_len` | the method's longest | 32768, 65536, 131072, 262144, 1048576, up to the method's `max_seq_len` |
| `val_dataset_id` | | a ready dataset of at most 5,000 samples; its loss is reported as `val_loss` |
| `eval_steps` | ~10 evaluations a run | steps between validation passes (at least 1: a 3-step run validates every step) |
| `seed` | 42 | |
| `max_grad_norm` | the model's (`defaults`) | 0 (no clipping) to 1e6. Most models train without clipping by default; 1.0 costs about 1% of the speed |
| `adam_beta1`, `adam_beta2`, `adam_epsilon` | the model's (0.9, 0.999, 1e-8 unless it sets them) | betas 0 to < 1, epsilon up to 1e-3 |
| `tokens_per_gpu` | the model's `max_tokens_per_gpu` | each GPU's packed row a micro-step: 8,192 up to `max_tokens_per_gpu`, and at least the dataset's longest sample. Not with `batch_size` |
| `gradient_accumulation_steps` | 1 | 1-64 micro-steps an optimizer step. Not with `batch_size` |
| `lora_rank`, `lora_alpha`, `lora_dropout` | 32, 64, the model's | LoRA only: rank 1-128, alpha 1-512, dropout 0-0.5 |
| `freeze_aligner` | true | `false` also trains the projector between the vision tower and the language model (in full, also beside LoRA adapters; the vision tower stays frozen). On the models that train on images, `full` and `lora` |
| `engine_debug_log` | false | also log the engine's setup details (kernels, memory layout, packing); for debugging |

`wandb={"api_key": ..., "project": ..., "entity": ..., "run_name": ...}` also logs to the owner's Weights & Biases
(the key is stored encrypted and never returned).

## Batch and steps

A step's size is set one of three ways: by default (each GPU packs `max_tokens_per_gpu` tokens, below); by
`batch_size` (samples a step); or by `tokens_per_gpu` with `gradient_accumulation_steps` (a step is about
`gpus x tokens_per_gpu x gradient_accumulation_steps` tokens, in whole samples). The last works where
`max_tokens_per_gpu` is set, including models without `batch_size`; the estimate's `steps` shows what it gives.

Without `batch_size`, each GPU packs about 64K tokens a step (128K for `qwen3.8-27b` and `gemma-4-26b-a4b`), so a step
on 8 GPUs trains roughly 470K tokens (about 1M on those two), whatever the sample lengths. Steps per epoch ≈ dataset
tokens (`dataset.tokens[model].total`) / tokens a step: 5M tokens on 8 GPUs ≈ 11 steps an epoch at 64K, 5 at 128K. `max_steps` cannot add steps beyond the epochs.

`batch_size` (samples a step, where `models()` allows it) gives smaller, more numerous steps, but each GPU must get at
least 8K tokens a step: the smallest batch is about 8 GPUs × 8,192 / average sample length (the server refuses less
and says the minimum). Below about 16K a step runs less efficiently (a notice says so): 6,000 short image samples at
8K tokens a GPU trained `qwen3.5-4b` at 6,200-8,700 tokens/s/GPU (12,000 listed) and `qwen3.6-35b-a3b` at
1,100-2,200 (11,500 listed); the cost per token is the same, the run takes longer. The job's notices report the
batch it actually used.

Decide with the steps the job will really have: `estimate` returns `steps` for the batch you give. The same data can be
under one step at the default batch and 17 steps at the smallest allowed `batch_size`.

### Small datasets

A dataset under one step's tokens (see above; for example a few thousand short support conversations) trains
**one step per epoch**, and `batch_size` cannot help when the minimum batch is larger than the dataset. Then:

- `num_train_epochs` (up to 20) is then the number of optimizer steps. Sweep it (for example 3 / 6 / 10) and pick by
  `val_loss`; with so few steps, warmup and the schedule hardly matter.
- Or add data: more examples, or several phrasings per example, usually helps more than more epochs.
- The run is cheap (cents), so the smoke run and the real run are the same job.

## Cost and time

- `jobs.estimate` returns what is **held** from the balance: estimated tokens × price. When the job ends it is charged
  for the tokens it actually trained, rounded up to whole cents, and the rest is released.
- **Every token of every sample is billed**: system, user and assistant messages plus the chat template's few tokens
  per message, not only the assistant tokens the loss is computed on. So `job.tokens_trained` is a little above
  `dataset.tokens[model].total` × epochs.
- The estimate counts with the largest token count over all models (`est_tokens`), so the charge can come out a little
  above or below it.
- With `layout="auto"` and `max_steps`, the hold is for the largest GPU setup the model may get. A setup picked by hand
  raises the price per token by 5% for each size above the model's smallest (Qwen on 16 GPUs: 1.05×, 32: 1.10×);
  `auto` never does.
- **A failed job is free**; a cancelled one pays for the tokens trained before it stopped. Validation passes are not
  charged. `metered: false` means the model has no price: the job is free.
- `estimate` runs the same checks as `create` and raises `APIError` (422) with the reason when the job would be
  refused. Otherwise it also returns `params` (the settings it would run with), `steps` (about how many optimizer
  steps; None for the large models with a tuned pack), `truncated_samples` and `warnings` (`FEW_STEPS`,
  `SAMPLES_TRUNCATED`).
- After a job ends, `jobs.billing(job_id)` gives what it held and was charged (`charged_credits`, US cents).
- Training time ≈ tokens / (`tokens_per_s_per_gpu` × GPUs), plus fixed time around it (medians of past jobs):

| model size | start-up | save | upload | checkpoint |
|---|---|---|---|---|
| `qwen3.8-27b`, `gemma-4-26b-a4b`, `muse-glimmer-30b` (and their long-context ids) | 2 min | 1-2 min | 6-20 min | 52-60 GB |
| `nemotron-3-super-120b`, `qwen3.8-next-180b` | 6-8 min | 3-6 min | 22-28 min | 250-360 GB |
| the `olmo-3-7b` ids, `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-9b-base` | 1.5-2 min | under 1 min | 2-5 min | 9-19 GB |
| `qwen3.6-35b-a3b`, `qwen3.5-35b-a3b-base` | 4 min | 1 min | about 45 min | 72 GB |
| `mimo-311b`, `glm-5.3-flash`, `minimax-m3.1-flash`, `deepseek-v4.1-flash` | 4-5 min | 9-13 min | 35-60 min | 600 GB+ |

  Plus the wait in the queue when no GPU setup is free (`models()` shows `available` and `queued`). Uploads from one
  cluster go one at a time, so an upload can wait behind other jobs'. The first step is slow (kernels compile); the
  steps after it show the real speed.

## Running experiments

- **Smoke run first** on new data or a new model, when the data is larger than a few steps: `max_steps: 3`, an
  explicit `layout` (the smallest in `methods[...].layouts`, so the estimate is not held for a larger setup) and,
  where the model takes one, the smallest `batch_size` (a too-small one gets a 422 naming the minimum). A default step
  is large (see Batch and steps), so 3 default steps can cost half an epoch. Every
  succeeded job also uploads a full checkpoint (tens of GB), smoke runs included. Its loss should be finite and in a normal range
  (about 0.3-3 for chat data). For small datasets see above: the real run is already small.
- **One change at a time** between runs; same data, same seed.
- **Validation set** on every real run, from the same distribution as the training data but not in it. Compare runs
  on `val_loss`, not training loss. Keep a separate **test set** that is never uploaded for the final comparison with
  the base model: the validation set steers your choices, so it is no longer an unbiased judge.
- **Sweeps:** an account can have 10 jobs queued or running. Submit the whole sweep (`nextron-cloud sweep`, below),
  then wait; jobs queue when GPUs are busy and run in submission order. `jobs.list()` returns every job of the
  organisation, other members' and older runs included, so find yours by tag.
- **A budget** means what is charged, but plan with the estimates (`credits`, what is held): the charge stays close
  to them and is usually a little lower or higher. Sum the estimates of every job you propose.
- **Learning rate** is the first thing to sweep: for example full 5e-6 / 1e-5 / 2e-5, LoRA 5e-5 / 1e-4 / 2e-4.
  Then epochs (1-3): stop increasing when `val_loss` turns up.
- **Too few steps** (under ~20) make warmup and the schedule coarse: set `batch_size` smaller or train more epochs.
- Name every job after what it tests, and give related jobs a tag; `report` then compares them without a separate
  log.
- Only the final step's checkpoint is kept. When the best `val_loss` is at an earlier step, train again with that
  many steps (or epochs) to get its weights.

A sweep is one job per combination of values; its jobs are named after their settings and tagged `sweep:<name>`:

```bash
nextron-cloud sweep --name support-lr --model qwen3.8-27b --dataset ds_... --layout 1x8-h100 \
    --param train_type=lora --param val_dataset_id=ds_... --grid learning_rate=5e-5,1e-4,2e-4 --grid num_train_epochs=2,4
# prints every job's estimate and the total (exit code 3); after the owner agrees, add --yes
nextron-cloud report --tag sweep:support-lr --table     # steps, loss, best val_loss and its step, tokens, minutes
```

From Python: `nextron_cloud.experiments.sweep(client, name=..., model=..., dataset_id=..., axes={...}, params={...},
start=False)` returns the estimates; `start=True` creates the jobs. Starting the same sweep again creates nothing new.
`experiments.report(client, tag=...)` returns the comparison rows.
