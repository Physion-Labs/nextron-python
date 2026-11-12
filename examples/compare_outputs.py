"""Answer the same prompts with a fine-tuned checkpoint and its base model, side by side.

Serve both with vLLM (or any OpenAI-compatible server) on a GPU machine first:

    vllm serve checkpoints/job_... --served-model-name ft --port 8000
    vllm serve <base model> --served-model-name base --port 8001

then, with a "messages" JSONL whose last message is the reference answer (for example val.jsonl):

    python examples/compare_outputs.py val.jsonl --ft http://localhost:8000/v1 --base http://localhost:8001/v1 --out compare.jsonl

Each output row has the prompt, the reference, both answers and whether each matches the reference exactly. Training
ran in non-thinking mode, so prompts are sent with enable_thinking false unless --thinking is given.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import httpx


def answer(client: httpx.Client, url: str, model: str, messages: list, args) -> str:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": args.thinking}}
    r = client.post(f"{url}/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def compare(client: httpx.Client, sample: dict, args) -> dict:
    *prompt, reference = sample["messages"]
    row = {"prompt": prompt, "reference": reference["content"]}
    for name, url, model in (("ft", args.ft, args.ft_model), ("base", args.base, args.base_model)):
        if url:
            row[name] = answer(client, url, model, prompt, args)
            row[f"{name}_exact"] = row[name].strip() == row["reference"].strip()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", help='a "messages" JSONL; the last message of each is the reference')
    parser.add_argument("--ft", required=True, help="the fine-tuned model's OpenAI-compatible base URL")
    parser.add_argument("--base", help="the base model's URL, to compare against")
    parser.add_argument("--ft-model", default="ft")
    parser.add_argument("--base-model", default="base")
    parser.add_argument("--out", default="compare.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-key", default="none", help="sent as a Bearer token to both servers")
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][: args.limit]
    rows, failed = [], 0
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(timeout=600, headers=headers) as client, ThreadPoolExecutor(args.workers) as pool, \
            open(args.out, "w", encoding="utf-8") as out:
        futures = [pool.submit(compare, client, s, args) for s in samples]
        for future in futures:   # each row is written as it comes, so a failure keeps the rest
            try:
                row = future.result()
            except httpx.HTTPError as error:
                failed += 1
                print(f"request failed: {error}")
                continue
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"samples": len(rows), "failed": failed, "out": args.out}
    for name in ("ft", "base"):
        if rows and name in rows[0]:
            summary[f"{name}_exact_match"] = round(sum(r[f"{name}_exact"] for r in rows) / len(rows), 4)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
