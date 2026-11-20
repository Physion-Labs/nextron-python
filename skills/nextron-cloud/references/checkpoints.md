# Checkpoints

## Download

The checkpoint is separate from training: it uploads after the job succeeds and stays for 30 days, so download it
when the weights are needed, not as part of every experiment.

```python
client.jobs.wait_checkpoint(job_id)          # returns at once if it is already uploaded
info = client.jobs.checkpoint(job_id)        # step, bytes, expires (unix time), files with 24-hour links
client.jobs.download_checkpoint(job_id, f"checkpoints/{job_id}")
```

- Check free disk space against `info.bytes` first: about 55 GB for a ~30B model, 250-360 GB for 120-180B, 600 GB and
  more for the largest.
- Every downloaded file's sha256 is checked against the checkpoint's list; a mismatch removes the file and raises.
- Files already complete are skipped and a partial file is resumed: if the download stops, run it again.
- `include=["*.json", "*.jinja", "tokenizer*"]` (CLI `--include`, repeatable) downloads only matching files, e.g. the
  config and tokenizer without the weights. No matching file is an error, not an empty download.
- Kept **30 days** after training; afterwards `checkpoint` raises 410. Links last 24 hours; call again for new ones.
- 409 means it is still uploading (`job.checkpoint["status"] == "uploading"`); wait.

## What it is

A Hugging Face model directory in bf16: `config.json`, `generation_config.json`, `model-*.safetensors` with
`model.safetensors.index.json`, the tokenizer files and `chat_template.jinja`. LoRA jobs are merged: the directory is
a full model. It loads wherever the base model loads, with the same versions.

## Serve and evaluate

vLLM (needs GPUs that hold the model; a version that supports the base model):

```bash
vllm serve checkpoints/job_... --served-model-name ft --max-model-len 32768 --tensor-parallel-size 1
```

```python
from openai import OpenAI
llm = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
reply = llm.chat.completions.create(
    model="ft", messages=[{"role": "user", "content": "..."}], temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},   # match training, see below
)
```

- **Thinking mode:** jobs train in non-thinking mode (data.md). For models with a thinking switch (Qwen), serve with
  `enable_thinking: False` unless the training data had `<think>` reasoning; otherwise the model is prompted
  differently from how it was trained.
- **Compare against the base model** on the same held-out prompts, served the same way: that is the effect of the
  fine-tune. Use prompts that were not in the training data. `examples/compare_outputs.py` in the SDK's repository
  (https://github.com/Physion-Labs/nextron-python/blob/main/examples/compare_outputs.py, not in the installed package)
  does it: serve both models, then `python examples/compare_outputs.py data/val.jsonl --ft http://localhost:8000/v1
  --base http://localhost:8001/v1` writes both answers next to the reference and prints exact-match rates.
- Use the task's own metric (exact match, a test suite, a rubric scored by a judge model) on a fixed evaluation set,
  and report the score with the job's name and id.
- SGLang (`python -m sglang.launch_server --model-path checkpoints/job_...`) and transformers
  (`AutoModelForCausalLM.from_pretrained`, or the base model's own class) work the same way.

No GPUs to serve it on: an inference run serves the job's model on NexCloud's GPUs for 7 days after training, if the
model lists `inference` in `client.models()[i].methods` ([inference.md](inference.md)). For a model without it, or
after the 7 days, say so to the owner; validation loss (`val_loss` in the job's metrics) is then the comparison
between runs.

## Verifying a checkpoint

`download_checkpoint` already checks each file against the checkpoint's sha256 list. Independently of the download
links, `jobs.manifest(job_id)` (once the checkpoint is uploaded) returns the job's manifest, JSON with `files` (record
files) and `checkpoint` (checkpoint files), each `{name: sha256}`; its own sha256 is `job.manifest_sha256`.
