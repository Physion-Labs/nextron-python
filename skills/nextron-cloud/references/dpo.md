# Preference training (DPO)

A DPO job trains on pairs of replies to the same prompt: one preferred (`chosen`), one not (`rejected`). The model
learns to make the preferred reply more likely than the rejected one, measured against the model it started from.
Full-parameter only. The models that train it list `dpo` in `client.models()[i].methods`; use the job's
`train_type="dpo"`.

## Data

One kind per dataset: preference pairs, in any of these forms.

```json
{"messages": [{"role": "user", "content": "..."}], "chosen": "a reply", "rejected": "another reply"}
{"prompt": "...", "chosen": {"role": "assistant", "content": "..."}, "rejected": {"role": "assistant", "content": "..."}}
{"chosen": [user, assistant], "rejected": [user, assistant]}
```

- The prompt ends with the user's turn; both replies are non-empty. The last form is two conversations that differ only
  in their last reply (UltraFeedback, HH-RLHF style).
- `check_file` reports `kind="preference"`; after upload `dataset.kind == "preference"` and
  `dataset.tokens[model]["max"]` is the longest pair: the prompt twice and both replies. It must be at most the method's
  `max_seq_len`.
- Pairs work best when both replies are close to what the model itself writes, for example two of its own replies ranked
  by a judge. Replies written by other models, especially when the chosen one is much longer, can leave the loss at or
  above 0.693.

## Settings

```python
job = client.jobs.create(model="qwen3.8-27b", dataset_id=ds.id,
                         params={"train_type": "dpo", "beta": 0.1, "learning_rate": 5e-7, "num_train_epochs": 1})
```

| Param | Default | |
|---|---|---|
| `beta` | 0.1 | how far the model may move from where it started: lower moves further |
| `learning_rate` | 5e-7 | DPO wants a much lower rate than SFT |
| `num_train_epochs` or `max_steps` | 1 epoch | |
| `warmup_ratio`, `weight_decay`, `lr_scheduler_type`, `seed` | as SFT | |

`batch_size`, `max_seq_len` and validation sets do not apply: a step packs whole pairs into each GPU's row, and
`client.jobs.plan` gives about how many steps a dataset makes.

## What to watch

Before step 1 the job scores every pair with the starting model (about a minute per few thousand pairs). The loss is
then exactly 0.693 (ln 2) at step 1 and falls as the model prefers the chosen replies; each step trains on pairs it has
not seen yet in that epoch, so a falling loss means the preference carries over to new pairs.
