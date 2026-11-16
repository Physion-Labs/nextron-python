"""Training data: the sample format, conversion from common formats, and the server's format check run locally.

A dataset is a JSONL file, one sample per line:

    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

At least one assistant message is needed, and a system message may only come first. These formats are converted to it
(here by `to_messages`, and by the server on upload):

    {"conversations" | "conversation": [{"from"|"role": ..., "value"|"content": ...}]}   ShareGPT, LMSYS
    {"instruction": ..., "input" | "context": ..., "output" | "response": ...}       Alpaca, Dolly
    {"prompt": ..., "completion" | "response": ...}                                  prompt / completion
    {"query" | "question": ..., "response" | "answer": ...}                          question / answer (OpenOrca)
    {"problem": ..., "solution": ...}, {"input": ..., "output": ...}                 problem / solution
    "content": [{"type": "text", "text": ...}, ...]                                   OpenAI content parts

An optional top-level "system" or "system_prompt" string becomes the system message.

A user message of a "messages" sample may carry images (JPEG or PNG, up to 16 a sample, 20 MB each), as content parts:

    {"role": "user", "content": [{"type": "image", "image": "photos/cat.jpg"}, {"type": "text", "text": "What is this?"}]}

An image is a file path (relative to the JSONL file) or a base64 "data:" URI; OpenAI's {"type": "image_url",
"image_url": {"url": ...}} is read the same way. A URL is not fetched: download the file first. Uploading packs the
image files with the samples (`write_bundle`).

Reinforcement learning (train_type "grpo") takes prompts with the answer the reward checks the model's reply against,
one kind per file:

    {"messages": [system?, user, ...], "answer": "..."}   the user's turn last
    {"prompt": "...", "answer": "..."}

A numeric answer is kept as its text, and an optional "metadata" value goes to a remote reward unchanged.

Preference training (train_type "dpo") takes a prompt with a preferred and a rejected reply:

    {"messages": [system?, user, ...], "chosen": reply, "rejected": reply}   a reply is a string or an assistant message
    {"prompt": "..." | [messages], "chosen": ..., "rejected": ...}             TRL's format
    {"chosen": [conversation], "rejected": [conversation]}                     two conversations that differ only in
                                                                               their last reply (UltraFeedback, HH-RLHF)
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
import tarfile
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Message = dict[str, str]

ROLES = ("system", "user", "assistant")
SHAREGPT_ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant",
                  "system": "system"}
PAIRS = (("prompt", "completion"), ("prompt", "response"), ("query", "response"), ("question", "answer"), ("question", "response"),
         ("problem", "solution"), ("input", "output"))
MEDIA_FIELDS = ("image", "images", "video", "videos", "audio", "audios")
MEDIA_ERROR = ("has an image, audio or video part; images go in the user messages of a \"messages\" sample "
               "(JPEG and PNG), audio and video are not trained on")
PLACEHOLDERS = ("<image>", "<video>", "<audio>")   # the trainer reads these in a sample's text as media
IMAGE_TYPES = {b"\xff\xd8\xff": "jpg", b"\x89PNG\r\n\x1a\n": "png"}
MAX_IMAGE_BYTES = 20 * 1000**2
MAX_IMAGE_PIXELS = 36_000_000
MAX_IMAGE_RATIO = 200   # long side / short side
JPEG_SIZE_MARKERS = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
MAX_IMAGES = 16   # per sample
MAX_LINE_BYTES = 32 << 20
MAX_FILE_BYTES = 5 * 1000**3
MAX_ERRORS = 5  # the server also reports only the first five


def _image(part: dict[str, Any]) -> Any:
    """What an image part names (a path, a data: URI or a bundle's "sha256:..."); None for a part of another type."""
    if part.get("type") == "image":
        return part.get("image")
    if part.get("type") == "image_url":
        url = part.get("image_url")
        return url.get("url") if isinstance(url, dict) else url
    return None


def _text(content: Any, images: list[str] | None = None) -> str:
    """A message's content as text. With `images`, an image part is "<image>" in the text and what it names is
    appended to the list; without, every part must be text."""
    if isinstance(content, str):
        parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
    elif isinstance(content, list) and all(isinstance(p, dict) for p in content):
        parts = content
    else:
        raise ValueError('every message needs a text "content"')
    out = []
    for part in parts:
        if part.get("type") == "text":
            piece = str(part.get("text", ""))
            if images is not None and any(tag in piece for tag in PLACEHOLDERS):
                raise ValueError("has <image>, <video> or <audio> written in its text, which a dataset with images "
                                 "cannot hold: the trainer reads them as media")
            out.append(piece)
            continue
        source = _image(part)
        if images is None or not isinstance(source, str):
            raise ValueError(MEDIA_ERROR)
        images.append(source)
        out.append("<image>")
    return "".join(out)


def read_image(source: str, base: Path) -> tuple[str, bytes]:
    """("jpg" | "png", the file's bytes) of an image a sample names: a path relative to `base`, or a data: URI."""
    if source.startswith("data:"):
        try:
            data = base64.b64decode(source.split(",", 1)[1], validate=True)
        except (IndexError, ValueError):
            raise ValueError("has an image whose data: URI is not base64") from None
    elif "://" in source:
        raise ValueError(f"names an image by URL ({source[:60]}); download it and give the file's path")
    else:
        try:
            data = (base / source).read_bytes()
        except OSError:
            raise ValueError(f"names an image file that cannot be read: {source[:120]}") from None
    kind = next((ext for magic, ext in IMAGE_TYPES.items() if data.startswith(magic)), None)
    if kind is None:
        raise ValueError(f"has an image that is neither JPEG nor PNG ({source[:60]}); convert it")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"has an image over {MAX_IMAGE_BYTES // 1000**2} MB ({source[:60]})")
    size = _png_size(data) if kind == "png" else _jpeg_size(data)
    if size is None:
        raise ValueError(f"has an image that is not a complete {'PNG' if kind == 'png' else 'JPEG'} file ({source[:60]})")
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"has an image with no pixels ({source[:60]})")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"has an image of more than {MAX_IMAGE_PIXELS // 10**6} million pixels ({width} x {height}, {source[:60]})")
    if max(width, height) > MAX_IMAGE_RATIO * min(width, height):
        raise ValueError(f"has an image more than {MAX_IMAGE_RATIO} times longer than wide ({width} x {height}, {source[:60]})")
    return kind, data


def _png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's header; None without one."""
    if data[12:16] != b"IHDR" or len(data) < 24:
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's segments before its image data; None when they are broken or give no size."""
    size = None
    at = 2
    while at + 4 <= len(data):
        if data[at] != 0xFF:
            return None
        marker = data[at + 1]
        if marker == 0xFF:   # fill byte
            at += 1
            continue
        if marker == 0xDA:   # the image data follows
            break
        length = struct.unpack(">H", data[at + 2:at + 4])[0]
        payload = data[at + 4:at + 2 + length]
        if size is None and marker in JPEG_SIZE_MARKERS and len(payload) >= 5:
            height, width = struct.unpack(">HH", payload[1:5])
            size = width, height
        at += 2 + length
    return size


def _image_parts(sample: Any) -> Iterator[tuple[list[Any], int, str]]:
    """(the content parts, the index, what the image names) of each image part in a sample's messages."""
    messages = sample.get("messages") if isinstance(sample, dict) else None
    for message in messages if isinstance(messages, list) else []:
        parts = message.get("content") if isinstance(message, dict) else None
        if not isinstance(parts, list):
            continue
        for i, part in enumerate(parts):
            source = _image(part) if isinstance(part, dict) else None
            if isinstance(source, str):
                yield parts, i, source


def to_messages(sample: Any, images: list[str] | None = None) -> list[Message]:
    """The sample as chat messages. ValueError says why it cannot be trained on. `images`: a list that takes what
    each image of the user messages names, in order ("<image>" stands for it in the text); without it images are refused."""
    if isinstance(sample, list):
        raise ValueError("is a JSON array; upload JSONL: one JSON object per line")
    if not isinstance(sample, dict):
        raise ValueError("is not a JSON object")
    if any(sample.get(f) for f in MEDIA_FIELDS):
        raise ValueError(MEDIA_ERROR)
    given = next((sample[k] for k in ("system", "system_prompt") if isinstance(sample.get(k), str) and sample[k]), None)
    system = [{"role": "system", "content": given}] if given else []
    if "messages" in sample:
        source = sample["messages"]
        if not isinstance(source, list) or not source:
            raise ValueError('needs a non-empty "messages" list')
        messages = []
        for m in source:
            if not isinstance(m, dict) or m.get("role") not in ROLES:
                raise ValueError('every message needs a "role" (system, user or assistant)')
            messages.append({"role": m["role"], "content": _text(m.get("content"), images if m["role"] == "user" else None)})
        if images and len(images) > MAX_IMAGES:
            raise ValueError(f"has {len(images)} images; a sample takes at most {MAX_IMAGES}")
    elif "conversations" in sample or "conversation" in sample:
        turns = sample.get("conversations", sample.get("conversation"))
        if not isinstance(turns, list) or not turns:
            raise ValueError('needs a non-empty "conversations" list')
        messages = []
        for t in turns:
            role = SHAREGPT_ROLES.get(str(t.get("from", t.get("role", ""))).lower()) if isinstance(t, dict) else None
            if role is None:
                raise ValueError('every turn needs "from" (human, gpt or system) and "value"')
            messages.append({"role": role, "content": _text(t.get("value", t.get("content")))})
    elif "instruction" in sample and ("output" in sample or "response" in sample):
        answer = "output" if "output" in sample else "response"
        prompt = "\n\n".join(str(sample[k]) for k in ("instruction", "input", "context") if sample.get(k))
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": _text(sample[answer])}]
    else:
        pair = next((p for p in PAIRS if p[0] in sample and p[1] in sample), None)
        if pair is None:
            raise ValueError('is not a format we know: use "messages", or one of conversations, instruction/output, '
                             "instruction/response, prompt/completion, query/response, question/answer, problem/solution, input/output")
        messages = [{"role": "user", "content": _text(sample[pair[0]])}, {"role": "assistant", "content": _text(sample[pair[1]])}]
    if not any(m["role"] == "assistant" and m["content"] for m in messages):
        raise ValueError("needs at least one non-empty assistant message to learn from")
    messages = system + messages
    if any(m["role"] == "system" for m in messages[1:]):
        raise ValueError("has a system message that is not the first message; a sample can have one, first")
    return messages


def is_rl(sample: Any) -> bool:
    """Whether the sample is a prompt with an answer (reinforcement learning) rather than a chat to imitate."""
    return isinstance(sample, dict) and "answer" in sample and ("messages" in sample or "prompt" in sample)


def to_rl(sample: dict[str, Any]) -> dict[str, Any]:
    """An RL sample as {"messages", "answer"}. ValueError says why it cannot be used."""
    answer = sample["answer"]
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):  # math sets often store numbers
        answer = str(answer)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('needs a non-empty "answer" string: what the reward compares the model\'s output with')
    if "messages" in sample:
        source = sample["messages"]
        if not isinstance(source, list) or not source or not isinstance(source[-1], dict) or source[-1].get("role") != "user":
            raise ValueError("an RL sample's messages end with the user's turn: the model writes the reply")
        chat = {**sample, "messages": source + [{"role": "assistant", "content": "-"}]}
    else:
        chat = {**sample, "completion": "-"}
    # Checked as a chat with a placeholder reply, so prompts follow the same rules as chat samples.
    row = {"messages": to_messages(chat)[:-1], "answer": answer}
    if "metadata" in sample:  # anything else a remote reward needs (test cases, rubric), passed to it unchanged
        row["metadata"] = sample["metadata"]
    return row


def is_preference(sample: Any) -> bool:
    """Whether the sample is a prompt with a preferred and a rejected reply (DPO)."""
    return isinstance(sample, dict) and "chosen" in sample and "rejected" in sample


def _reply(value: Any, name: str) -> Message:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict):
        if value.get("role") != "assistant":
            raise ValueError(f'"{name}" must be the assistant\'s reply')
        value = value.get("content")
    content = _text(value)
    if not content.strip():
        raise ValueError(f'needs a non-empty "{name}" reply')
    return {"role": "assistant", "content": content}


def to_preference(sample: dict[str, Any]) -> dict[str, Any]:
    """A DPO sample as {"messages", "chosen", "rejected"}. ValueError says why it cannot be used."""
    chosen, rejected = sample["chosen"], sample["rejected"]
    if "messages" in sample or "prompt" in sample:
        prompt = sample.get("messages", sample.get("prompt"))
        prompt = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    elif isinstance(chosen, list) and isinstance(rejected, list) and len(chosen) > 1 and chosen[:-1] == rejected[:-1]:
        prompt, chosen, rejected = chosen[:-1], chosen[-1], rejected[-1]
    else:
        raise ValueError('a preference sample needs "messages" or "prompt" with "chosen" and "rejected", or two '
                         '"chosen" / "rejected" conversations that differ only in their last reply')
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[-1], dict) or prompt[-1].get("role") != "user":
        raise ValueError("a preference sample's prompt ends with the user's turn: the replies answer it")
    chosen, rejected = _reply(chosen, "chosen"), _reply(rejected, "rejected")
    messages = to_messages({**sample, "messages": prompt + [chosen]})  # the prompt follows the same rules as chat samples
    return {"messages": messages[:-1], "chosen": chosen, "rejected": rejected}


def to_sample(sample: Any, images: list[str] | None = None) -> dict[str, Any]:
    """The sample as the server stores it: {"messages"} for a chat, {"messages", "answer"} for RL, {"messages",
    "chosen", "rejected"} for DPO. `images`: as in `to_messages`, for a chat."""
    if is_rl(sample):
        return to_rl(sample)
    if is_preference(sample):
        return to_preference(sample)
    return {"messages": to_messages(sample, images)}


def _line(sample: Any) -> Any:
    """The sample as a line of a JSONL file: in the server's form, or as it is when it has images (they stay content parts)."""
    images: list[str] = []
    row = to_sample(sample, images)
    return sample if images else row


def write_jsonl(samples: Iterable[Any], path: str | os.PathLike[str]) -> int:
    """Write samples in any format above as JSONL in the server's form; returns how many. Raises on the first bad one.
    A sample with images is written as it is; its image paths are read relative to `path`'s folder."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as out:
        for number, sample in enumerate(samples, 1):
            try:
                row = _line(sample)
            except ValueError as error:
                raise ValueError(f"sample {number}: {error}") from None
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


@dataclass
class CheckResult:
    samples: int = 0
    chars: int = 0  # message text; a token is roughly 3-4 characters of English
    errors: list[str] = field(default_factory=list)
    kind: str | None = None  # "chat", "rl" for prompts with answers, or "preference" for DPO pairs
    images: int = 0  # image parts in the samples; with any, the upload is a bundle (write_bundle)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_file(path: str | os.PathLike[str]) -> CheckResult:
    """Run the server's format check on a local JSONL file. Token counts per model come from the server after upload."""
    path = Path(path)
    result = CheckResult()
    if path.stat().st_size > MAX_FILE_BYTES:
        result.errors.append(f"the file is over {MAX_FILE_BYTES // 1000**3} GB")
        return result
    kinds = set()
    with path.open("rb") as f:
        for number, raw in enumerate(f, 1):
            if len(raw) > MAX_LINE_BYTES:
                _note(result, f"line {number}: longer than {MAX_LINE_BYTES >> 20} MB")
                continue
            if number == 1:
                raw = raw.removeprefix(b"\xef\xbb\xbf")
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                _note(result, f"line {number}: not UTF-8 text; save the file as UTF-8")
                continue
            if not line:
                continue
            try:
                found: list[str] = []
                row = to_sample(json.loads(line), found)
                for source in found:
                    read_image(source, path.parent)
            except json.JSONDecodeError as error:
                _note(result, f"line {number}: not valid JSON ({error.msg})")
                continue
            except ValueError as error:
                _note(result, f"line {number}: {error}")
                continue
            result.samples += 1
            result.images += len(found)
            replies = [row[k] for k in ("chosen", "rejected") if k in row]
            result.chars += sum(len(m["content"]) for m in row["messages"] + replies)
            kinds.add("rl" if "answer" in row else "preference" if "chosen" in row else "chat")
    if not result.errors and not result.samples:
        result.errors.append("the file has no samples")
    if len(kinds) > 1:
        result.errors.append("the file mixes chat samples, prompts with answers (RL) and preference pairs; a dataset holds one kind")
    result.kind = kinds.pop() if len(kinds) == 1 else None
    return result


def write_bundle(path: str | os.PathLike[str], out: str | os.PathLike[str]) -> int:
    """Pack a JSONL file whose samples carry images into the tar the server takes: each image once, as
    images/<SHA-256 of its bytes>.jpg|png, then the samples as data.jsonl with every image named "sha256:...".
    Returns the number of image files. ValueError names the first line that cannot be packed."""
    path = Path(path)
    packed: set[str] = set()
    with tarfile.open(out, "w") as tar, tempfile.TemporaryFile() as samples:
        def add(name: str, size: int, file: Any) -> None:
            info = tarfile.TarInfo(name)
            info.size = size
            tar.addfile(info, file)

        with path.open("rb") as f:
            for number, raw in enumerate(f, 1):
                line = raw.removeprefix(b"\xef\xbb\xbf").decode("utf-8").strip()
                if not line:
                    continue
                sample = json.loads(line)
                for parts, i, source in _image_parts(sample):
                    try:
                        kind, data = read_image(source, path.parent)
                    except ValueError as error:
                        raise ValueError(f"line {number}: {error}") from None
                    sha = hashlib.sha256(data).hexdigest()
                    if sha not in packed:
                        packed.add(sha)
                        add(f"images/{sha}.{kind}", len(data), io.BytesIO(data))
                    parts[i] = {"type": "image", "image": f"sha256:{sha}"}
                samples.write(json.dumps(sample, ensure_ascii=False).encode() + b"\n")
        size = samples.tell()
        samples.seek(0)
        add("data.jsonl", size, samples)
    return len(packed)


def _note(result: CheckResult, error: str) -> None:
    if len(result.errors) < MAX_ERRORS:
        result.errors.append(error)


VAL_MAX_SAMPLES = 5000  # the most a validation set may have


def read_rows(path: str | os.PathLike[str]) -> list[Any]:
    """Rows of a .jsonl, .json (a list) or .csv file, as parsed values (CSV rows as dicts of strings)."""
    path = Path(path)
    if path.suffix == ".csv":
        import csv
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path.name} is not a JSON list of samples")
        return rows
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_file(path: str | os.PathLike[str], out_dir: str | os.PathLike[str] = ".", *, val_fraction: float = 0.1,
                 seed: int = 0, prompt_column: str | list[str] | None = None, response_column: str | None = None,
                 answer_column: str | None = None, system: str | None = None) -> dict[str, Any]:
    """Convert a file to train.jsonl and val.jsonl in `out_dir`: rows in any accepted format, or user and assistant
    columns (`prompt_column`, several joined by a blank line when non-empty; `response_column`), e.g. of a CSV. With
    `answer_column` instead of `response_column`, the rows become RL samples (prompt + answer). Rows that cannot be
    trained on are left out and counted. Samples with images keep them, their paths rewritten for `out_dir`. Shuffled
    with `seed`; the validation set is `val_fraction` of the rows, at most 5,000."""
    import random

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be at least 0 and below 1")
    rows = read_rows(path)
    prompt_columns = [prompt_column] if isinstance(prompt_column, str) else prompt_column or []
    target_column = response_column or answer_column
    if prompt_columns and target_column and rows:
        missing = [c for c in (*prompt_columns, target_column) if c not in rows[0]]
        if missing:
            raise ValueError(f"{Path(path).name} has no column {', '.join(missing)}; it has {', '.join(rows[0])}")
    samples: list[Any] = []
    errors: list[str] = []
    skipped = 0
    out = Path(out_dir)
    for number, row in enumerate(rows, 1):
        if prompt_columns and target_column:
            prompt = "\n\n".join(str(row[c]) for c in prompt_columns if row[c])
            row = {"prompt": prompt, "completion" if response_column else "answer": row[target_column]}
        if system:
            row = dict(row, system=system)
        try:
            line = _line(row)
            for parts, i, source in _image_parts(line):
                if not source.startswith("data:") and "://" not in source:   # a path: train.jsonl may be in another folder
                    parts[i] = {"type": "image", "image": os.path.relpath(Path(path).parent / source, out)}
            samples.append(line)
        except ValueError as error:
            skipped += 1
            if len(errors) < MAX_ERRORS:
                errors.append(f"row {number}: {error}")
    random.Random(seed).shuffle(samples)
    n_val = min(round(len(samples) * val_fraction), VAL_MAX_SAMPLES) if val_fraction else 0
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"skipped": skipped, "errors": errors}
    for name, part in (("train", samples[n_val:]), ("val", samples[:n_val])):
        target = out / f"{name}.jsonl"
        target.unlink(missing_ok=True)  # never leave an earlier run's file behind to be uploaded by mistake
        if part:
            write_jsonl(part, target)
            report[name] = {"path": str(target), "samples": len(part)}
    return report
