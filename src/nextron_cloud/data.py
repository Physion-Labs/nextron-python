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

A user message of a "messages" sample may carry images (JPEG or PNG, up to 16 a sample, 20 MB each) and videos (MP4,
up to 4 a sample, 500 MB each), as content parts:

    {"role": "user", "content": [{"type": "image", "image": "photos/cat.jpg"}, {"type": "text", "text": "What is this?"}]}
    {"role": "user", "content": [{"type": "video", "video": "clips/race.mp4"}, {"type": "text", "text": "Who wins?"}]}

An image is a file path (relative to the JSONL file) or a base64 "data:" URI; OpenAI's {"type": "image_url",
"image_url": {"url": ...}} is read the same way. A video is a file path. A URL is not fetched: download the file first.
Uploading packs the image and video files with the samples (`write_bundle`).

A chat sample may use tools, in OpenAI's format: the functions the model may call in a top-level "tools" list, an
assistant message's calls in "tool_calls" ({"id", "type": "function", "function": {"name", "arguments"}}, arguments an
object or a JSON string), and each result in a {"role": "tool", "tool_call_id", "content"} message:

    {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}],
     "messages": [{"role": "user", "content": "Weather in Paris?"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                      "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "18C"},
                  {"role": "assistant", "content": "It is 18C."}]}

Ai2 / Dolci tool rows ("functions", "function_calls", "environment"), ShareGPT tool rows ("function_call" and
"observation" turns) and Hermes rows (<tool_call> JSON in the assistant's text) are converted to it (to_tool_chat).
Training renders each model's own tool format and puts loss on the assistant's calls and answers, never on results.

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

import ast
import base64
import hashlib
import io
import json
import os
import re
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
MEDIA_ERROR = ("has an image, video or audio part; images and videos go in the user messages of a \"messages\" "
               "sample (JPEG, PNG and MP4), audio is not trained on")
PLACEHOLDERS = ("<image>", "<video>", "<audio>")   # the trainer reads these in a sample's text as media
IMAGE_TYPES = {b"\xff\xd8\xff": "jpg", b"\x89PNG\r\n\x1a\n": "png"}
MAX_IMAGE_BYTES = 20 * 1000**2
MAX_IMAGE_PIXELS = 36_000_000
MAX_IMAGE_RATIO = 200   # long side / short side
JPEG_SIZE_MARKERS = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
MAX_IMAGES = 16   # per sample
MAX_VIDEO_BYTES = 500 * 1000**2
MAX_VIDEO_SECONDS = 60
MIN_VIDEO_SECONDS = 1.0
MAX_VIDEO_PIXELS = 3840 * 2160   # a frame, as uploaded
MP4_BRANDS = {b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"M4V ", b"qt  ", b"dash"}
MAX_VIDEOS = 4   # per sample
MAX_LINE_BYTES = 32 << 20
MAX_FILE_BYTES = 5 * 1000**3
BUNDLE_MAX_BYTES = 20 * 1000**3   # a file with images or videos is uploaded as one tar of the samples and the files
BUNDLE_MAX_IMAGES = 200_000   # image and video files
IMAGE_REF_CHARS = len("sha256:") + 64   # how a bundle names an image or a video
MAX_ERRORS = 5  # the server also reports only the first five


class _Tagged(ValueError):
    """A media tag written in a sample's text. The server refuses it in a dataset with images or videos only, so in a
    sample without them it is an error only when another sample of the file has one."""


def _media(part: dict[str, Any]) -> tuple[str, Any] | None:
    """("image" | "video", what the part names: a path, a data: URI or a bundle's "sha256:..."); None for a part of
    another type."""
    if part.get("type") in ("image", "video"):
        return part["type"], part.get(part["type"])
    if part.get("type") == "image_url":
        url = part.get("image_url")
        return "image", url.get("url") if isinstance(url, dict) else url
    return None


def _text(content: Any, images: list[str] | None = None, tags: bool = False) -> str:
    """A message's content as text. With `images`, an image or video part is "<image>" or "<video>" in the text and
    what it names is appended to the list; without, every part must be text. With `tags`, a media tag written in the
    text is refused."""
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
            if tags and any(tag in piece for tag in PLACEHOLDERS):
                raise _Tagged("has <image>, <video> or <audio> written in its text, which a dataset with images or "
                              "videos cannot hold: the trainer reads them as media")
            out.append(piece)
            continue
        media = _media(part)
        if images is None or media is None or not isinstance(media[1], str):
            raise ValueError(MEDIA_ERROR)
        images.append(media[1])
        out.append(f"<{media[0]}>")
    return "".join(out)


def read_image(source: str, base: Path) -> tuple[str, bytes]:
    """("jpg" | "png", the file's bytes) of an image a sample names: a path relative to `base`, or a data: URI."""
    if source.startswith("data:"):
        try:
            data = base64.b64decode(source.split(",", 1)[1], validate=True)
        except (IndexError, ValueError):
            raise ValueError("has an image whose data: URI is not base64") from None
    else:
        data = _read_file(source, base)
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


def read_video(source: str, base: Path) -> tuple[str, bytes]:
    """("mp4", the file's bytes) of a video a sample names by a path relative to `base`, checked from its boxes."""
    if source.startswith("data:"):
        raise ValueError("has a video given as a data: URI; save it as an MP4 file and give the file's path")
    data = _read_file(source, base)
    try:
        mp4_header(data)
    except (struct.error, StopIteration, ZeroDivisionError):   # a box cut short, missing, or a zero time scale
        raise ValueError(f"has a video that is not a complete MP4 file ({source[:60]})") from None
    except ValueError as error:
        raise ValueError(f"has a video that {error} ({source[:60]})") from None
    return "mp4", data


def read_media(kind: str, source: str, base: Path) -> tuple[str, bytes]:
    """(file extension, bytes) of an image or a video (`kind` "image" or "video") a sample names."""
    return read_video(source, base) if kind == "video" else read_image(source, base)


def _read_file(source: str, base: Path) -> bytes:
    if "://" in source:
        raise ValueError(f"names a file by URL ({source[:60]}); download it and give the file's path")
    try:
        return (base / source).read_bytes()
    except OSError:
        raise ValueError(f"names a file that cannot be read: {source[:120]}") from None


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(type, payload start, payload end) of each MP4 box in data[start:end]."""
    at = start
    while at + 8 <= end:
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", data[at + 8:at + 16])[0], 16
        elif size == 0:
            size = end - at
        if size < head or at + size > end:
            raise ValueError("is not a complete MP4 file")
        yield kind, at + head, at + size
        at += size


def _find(data: bytes, start: int, end: int, path: list[bytes]) -> Iterator[tuple[int, int]]:
    """(payload start, payload end) of the boxes at the end of `path` (box types) under data[start:end]."""
    for kind, payload, stop in _boxes(data, start, end):
        if kind == path[0] and len(path) == 1:
            yield payload, stop
        elif kind == path[0]:
            yield from _find(data, payload, stop, path[1:])


def mp4_header(data: bytes) -> tuple[int, int, float, float]:
    """(width, height, seconds, frames a second) of an MP4 file's first video track, read from its boxes. ValueError
    says why the server would refuse the file."""
    if len(data) > MAX_VIDEO_BYTES:
        raise ValueError(f"is over {MAX_VIDEO_BYTES // 1000**2} MB")
    if data[4:8] != b"ftyp" or data[8:12] not in MP4_BRANDS:
        raise ValueError("is not an MP4 file")
    moov = next(_find(data, 0, len(data), [b"moov"]), None)
    if moov is None:
        raise ValueError("is not a complete MP4 file")
    for trak in _find(data, *moov, [b"trak"]):
        handler = next(_find(data, *trak, [b"mdia", b"hdlr"]), None)
        if handler is None or data[handler[0] + 8:handler[0] + 12] != b"vide":
            continue
        tkhd, end = next(_find(data, *trak, [b"tkhd"]))
        width, height = (value >> 16 for value in struct.unpack(">II", data[end - 8:end]))
        matrix = tkhd + (52 if data[tkhd] == 1 else 40)
        a, b = struct.unpack(">ii", data[matrix:matrix + 8])
        if a == 0 and b != 0:   # turned a quarter
            width, height = height, width
        mdhd, _ = next(_find(data, *trak, [b"mdia", b"mdhd"]))
        if data[mdhd] == 1:
            timescale, duration = struct.unpack(">IQ", data[mdhd + 20:mdhd + 32])
        else:
            timescale, duration = struct.unpack(">II", data[mdhd + 12:mdhd + 20])
        stsz, _ = next(_find(data, *trak, [b"mdia", b"minf", b"stbl", b"stsz"]))
        frames = struct.unpack(">I", data[stsz + 8:stsz + 12])[0]
        seconds = duration / timescale
        if not frames:
            raise ValueError("has no frames")
        if seconds < MIN_VIDEO_SECONDS:
            raise ValueError("is shorter than a second")
        if seconds > MAX_VIDEO_SECONDS:
            raise ValueError(f"is longer than {MAX_VIDEO_SECONDS} seconds")
        if width * height > MAX_VIDEO_PIXELS:
            raise ValueError("is larger than 3840 x 2160 a frame")
        if max(width, height) > MAX_IMAGE_RATIO * min(width, height):
            raise ValueError(f"is more than {MAX_IMAGE_RATIO} times longer than wide ({width} x {height})")
        return width, height, seconds, frames / seconds
    raise ValueError("has no video track")


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


def _media_parts(sample: Any) -> Iterator[tuple[list[Any], int, str, str]]:
    """(the content parts, the index, "image" | "video", what it names) of each image and video part in a sample's
    messages."""
    messages = sample.get("messages") if isinstance(sample, dict) else None
    for message in messages if isinstance(messages, list) else []:
        parts = message.get("content") if isinstance(message, dict) else None
        if not isinstance(parts, list):
            continue
        for i, part in enumerate(parts):
            media = _media(part) if isinstance(part, dict) else None
            if media is not None and isinstance(media[1], str):
                yield parts, i, media[0], media[1]


def is_tool_sample(sample: Any) -> bool:
    """Whether the sample has tools, tool calls or tool results in any of the formats to_tool_chat takes."""
    if not isinstance(sample, dict):
        return False
    if sample.get("tools") or sample.get("functions"):
        return True
    turns = sample.get("messages") or sample.get("conversations") or sample.get("conversation") or []
    for m in turns if isinstance(turns, list) else []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", m.get("from", ""))).lower()
        if role in TOOL_SOURCE_ROLES or any(m.get(k) for k in ("tool_calls", "function_call", "function_calls", "functions")):
            return True
    return False


def tool_spec(tool: Any) -> dict[str, Any]:
    """A tool definition as OpenAI's {"type": "function", "function": {"name", "description"?, "parameters"?}}."""
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        tool = tool["function"]
    if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
        raise ValueError('every tool needs a "name" (OpenAI: {"type": "function", "function": {"name", "parameters"}})')
    return {"type": "function", "function": tool}


def tool_list(value: Any) -> list[dict[str, Any]]:
    """A sample's tools: a list, or a JSON string of one (LLaMA-Factory, Ai2)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError('"tools" is a string that is not JSON') from None
    value = [value] if isinstance(value, dict) else value
    if not isinstance(value, list):
        raise ValueError('"tools" must be a list of function definitions')
    return [tool_spec(t) for t in value]


def arguments(value: Any) -> dict[str, Any]:
    """A call's arguments as an object: given as one, or as a JSON string of one (OpenAI's API sends a string)."""
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            raise ValueError("has tool call arguments that are not JSON") from None
    if not isinstance(value, dict):
        raise ValueError("has tool call arguments that are not a JSON object")
    return value


def python_calls(source: str) -> list[tuple[str, dict[str, Any]]]:
    """Ai2's function_calls string, one Python call per line (get_weather(city="Paris")), as (name, arguments)."""
    try:
        tree = ast.parse(source.strip(), mode="exec")
    except SyntaxError:
        raise ValueError(f"has function_calls that are not Python calls: {source[:80]!r}") from None
    calls = []
    for statement in tree.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (not isinstance(call, ast.Call) or call.args or any(k.arg is None for k in call.keywords)
                or not isinstance(call.func, (ast.Name, ast.Attribute))):
            raise ValueError(f"has function_calls that are not name(key=value, ...) calls: {source[:80]!r}")
        try:
            kwargs = {str(k.arg): ast.literal_eval(k.value) for k in call.keywords}
        except ValueError:
            raise ValueError(f"has function_calls whose arguments are not literal values: {source[:80]!r}") from None
        calls.append((ast.unparse(call.func), kwargs))
    return calls


HERMES_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
HERMES_TOOLS = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)
HERMES_RESPONSE = re.compile(r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.S)
TOOL_SOURCE_ROLES = ("tool", "environment", "function", "observation", "function_call", "ipython")


def to_tool_chat(sample: dict[str, Any]) -> dict[str, Any]:
    """A chat sample with tools as the server stores it, {"messages", "tools"} in OpenAI's format:
    tools at the top level; an assistant message's calls in "tool_calls" ({"id", "type": "function", "function": {"name",
    "arguments": object}}); each result a {"role": "tool", "tool_call_id", "content"} message. Taken as well:
        Ai2 / Dolci: "functions" (JSON) on the system message, "function_calls" (Python calls) on assistant messages,
            results in "environment" messages; the original messages are kept as "ai2" (Olmo is trained on them as written)
        ShareGPT (LLaMA-Factory): "function_call" turns (JSON {"name", "arguments"}) and "observation" turns, "tools"
        Hermes: <tool_call>{"name", "arguments"}</tool_call> in assistant text, tools in the system's <tools>...</tools>
    ValueError says why it cannot be used."""
    tools = tool_list(sample["tools"]) if sample.get("tools") else []
    name, ai2 = "messages + tools", None
    if "messages" in sample:
        source = sample["messages"]
        if not isinstance(source, list) or not source or not all(isinstance(m, dict) for m in source):
            raise ValueError('needs a non-empty "messages" list')
        if any("functions" in m or "function_calls" in m or m.get("role") == "environment" for m in source):
            name, ai2 = "Ai2 tools", [{k: m[k] for k in ("role", "content", "functions", "function_calls") if k in m} for m in source]
    else:
        turns = sample.get("conversations", sample.get("conversation"))
        if not isinstance(turns, list) or not turns or not all(isinstance(t, dict) for t in turns):
            raise ValueError('needs a non-empty "messages" (or "conversations") list')
        roles = SHAREGPT_ROLES | {r: r for r in TOOL_SOURCE_ROLES}
        source = [{"role": roles.get(str(t.get("from", t.get("role", ""))).lower(), "?"), "content": t.get("value", t.get("content"))}
                  for t in turns]
        name = "ShareGPT tools"
    given = next((sample[k] for k in ("system", "system_prompt") if isinstance(sample.get(k), str) and sample[k]), None)
    if given:
        source = [{"role": "system", "content": given}] + source
    messages: list[dict[str, Any]] = []
    calls_open: list[str] = []   # the last assistant message's call ids
    pending: list[str] = []   # ... not answered yet
    for m in source:
        role = m.get("role")
        if role == "system":
            content = _text(m.get("content"))
            hermes = HERMES_TOOLS.search(content)
            if m.get("functions"):   # Ai2: a JSON list after the system text
                tools += tool_list(m["functions"])
            elif not tools and hermes:
                found = hermes[1]
                try:
                    tools = tool_list(json.loads(found) if found.startswith("[") else [json.loads(line) for line in found.splitlines() if line.strip()])
                except json.JSONDecodeError:
                    raise ValueError("has <tools> in its system message that are not JSON") from None
                name = "Hermes tools"
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": _text(m.get("content"))})
            calls_open = pending = []
        elif role in ("assistant", "function_call"):
            content = _text(m.get("content") or "") if role == "assistant" else ""
            calls: list[tuple[Any, ...]] = []   # (name, arguments, id?)
            if role == "function_call":   # LLaMA-Factory: the turn is the call (or a list of them), as JSON
                found = m.get("content")
                try:
                    found = json.loads(found) if isinstance(found, str) else found
                except json.JSONDecodeError:
                    raise ValueError("has a function_call turn that is not JSON") from None
                for call in found if isinstance(found, list) else [found]:
                    if not isinstance(call, dict):
                        raise ValueError('a function_call turn needs {"name", "arguments"}')
                    calls.append((call.get("name"), call.get("arguments")))
            elif m.get("function_calls"):
                calls = python_calls(m["function_calls"])
            elif m.get("function_call"):   # OpenAI's old single call
                calls = [(m["function_call"].get("name"), m["function_call"].get("arguments"))]
            elif m.get("tool_calls"):
                for call in m["tool_calls"]:
                    function = call.get("function", call) if isinstance(call, dict) else None
                    if not isinstance(function, dict):
                        raise ValueError('every tool call needs a "function" with "name" and "arguments"')
                    calls.append((function.get("name"), function.get("arguments"), call.get("id")))
            elif "<tool_call>" in content and tools:
                for found in HERMES_CALL.findall(content):
                    try:
                        call = json.loads(found)
                    except json.JSONDecodeError:
                        raise ValueError("has a <tool_call> that is not JSON") from None
                    calls.append((call.get("name"), call.get("arguments", call.get("parameters"))))
                content = HERMES_CALL.sub("", content).strip()
                name = "Hermes tools"
            if role == "function_call" and messages and messages[-1]["role"] == "assistant" and "tool_calls" not in messages[-1]:
                message: dict[str, Any] = messages.pop()   # its text came in the turn before
            else:
                message = {"role": "assistant", "content": content}
            if calls:
                message["tool_calls"] = []
                for n, call in enumerate(calls):
                    function, args = call[0], call[1]
                    if not isinstance(function, str) or not function:
                        raise ValueError("has a tool call without a function name")
                    if function not in {t["function"]["name"] for t in tools}:
                        raise ValueError(f'calls {function!r}, which is not in the sample\'s tools')
                    call_id = (call[2] if len(call) > 2 else None) or f"call_{len(messages)}_{n}"
                    message["tool_calls"].append({"id": str(call_id), "type": "function", "function": {"name": function, "arguments": arguments(args)}})
                calls_open = pending = [c["id"] for c in message["tool_calls"]]
            messages.append(message)
        elif role in TOOL_SOURCE_ROLES:
            if not messages or (messages[-1]["role"] != "tool" and "tool_calls" not in messages[-1]):
                raise ValueError("has a tool result that does not follow the assistant's tool call")
            content = m.get("content")
            parts = isinstance(content, list) and all(isinstance(p, dict) and "type" in p for p in content)
            if not isinstance(content, str) and not parts:   # a result given as JSON
                content = json.dumps(content, ensure_ascii=False)
            content = HERMES_RESPONSE.sub(r"\1", _text(content))
            call_id = m.get("tool_call_id") or (pending[0] if pending else None)
            if call_id not in calls_open:
                raise ValueError("has more tool results than tool calls, or one whose tool_call_id matches no call")
            pending = [c for c in pending if c != call_id]
            messages.append({"role": "tool", "tool_call_id": str(call_id), "content": content})
        else:
            raise ValueError('every message needs a "role": system, user, assistant or tool')
    if any(m["role"] == "system" for m in messages[1:]):
        raise ValueError("has a system message that is not the first message; a sample can have one, first")
    if not any(m["role"] == "assistant" and (m["content"] or m.get("tool_calls")) for m in messages):
        raise ValueError("needs at least one assistant message with text or tool calls to learn from")
    if not tools:
        raise ValueError('has tool calls or results but no "tools": list the functions the model may call')
    row = {"messages": messages, "tools": tools}
    if ai2:
        row["ai2"] = ai2
    return row


def to_messages(sample: Any, images: list[str] | None = None) -> list[Message]:
    """The sample as chat messages. ValueError says why it cannot be trained on. `images`: a list that takes what
    each image and video of the user messages names, in order ("<image>" or "<video>" stands for it in the text);
    without it images and videos are refused."""
    if isinstance(sample, list):
        raise ValueError("is a JSON array; upload JSONL: one JSON object per line")
    if not isinstance(sample, dict):
        raise ValueError("is not a JSON object")
    if is_tool_sample(sample):   # chat samples with tools go through to_tool_chat
        raise ValueError("has tools or tool calls, which only chat (SFT) samples take, not RL prompts or preference pairs")
    if any(sample.get(f) for f in MEDIA_FIELDS):
        raise ValueError(MEDIA_ERROR)
    tags = images is not None   # the trainer reads a media tag anywhere in a sample of a file with images
    given = next((sample[k] for k in ("system", "system_prompt") if isinstance(sample.get(k), str) and sample[k]), None)
    system = [{"role": "system", "content": _text(given, tags=tags)}] if given else []
    if "messages" in sample:
        source = sample["messages"]
        if not isinstance(source, list) or not source:
            raise ValueError('needs a non-empty "messages" list')
        messages = []
        for m in source:
            if not isinstance(m, dict) or m.get("role") not in ROLES:
                raise ValueError('every message needs a "role" (system, user or assistant)')
            content = _text(m.get("content"), images if m["role"] == "user" else None, tags)
            messages.append({"role": m["role"], "content": content})
        if images:
            videos = [kind for _, _, kind, _ in _media_parts(sample)].count("video")
            if len(images) - videos > MAX_IMAGES:
                raise ValueError(f"has {len(images) - videos} images; a sample takes at most {MAX_IMAGES}")
            if videos > MAX_VIDEOS:
                raise ValueError(f"has {videos} videos; a sample takes at most {MAX_VIDEOS}")
    elif "conversations" in sample or "conversation" in sample:
        turns = sample.get("conversations", sample.get("conversation"))
        if not isinstance(turns, list) or not turns:
            raise ValueError('needs a non-empty "conversations" list')
        messages = []
        for t in turns:
            role = SHAREGPT_ROLES.get(str(t.get("from", t.get("role", ""))).lower()) if isinstance(t, dict) else None
            if role is None:
                raise ValueError('every turn needs "from" (human, gpt or system) and "value"')
            messages.append({"role": role, "content": _text(t.get("value", t.get("content")), tags=tags)})
    elif "instruction" in sample and ("output" in sample or "response" in sample):
        answer = "output" if "output" in sample else "response"
        prompt = "\n\n".join(str(sample[k]) for k in ("instruction", "input", "context") if sample.get(k))
        messages = [{"role": "user", "content": _text(prompt, tags=tags)},
                    {"role": "assistant", "content": _text(sample[answer], tags=tags)}]
    else:
        pair = next((p for p in PAIRS if p[0] in sample and p[1] in sample), None)
        if pair is None:
            raise ValueError('is not a format we know: use "messages", or one of conversations, instruction/output, '
                             "instruction/response, prompt/completion, query/response, question/answer, problem/solution, input/output")
        messages = [{"role": "user", "content": _text(sample[pair[0]], tags=tags)},
                    {"role": "assistant", "content": _text(sample[pair[1]], tags=tags)}]
    if not any(m["role"] == "assistant" and m["content"] for m in messages):
        raise ValueError("needs at least one non-empty assistant message to learn from")
    messages = system + messages
    if any(m["role"] == "system" for m in messages[1:]):
        raise ValueError("has a system message that is not the first message; a sample can have one, first")
    return messages


def is_rl(sample: Any) -> bool:
    """Whether the sample is a prompt with an answer (reinforcement learning) rather than a chat to imitate."""
    return isinstance(sample, dict) and "answer" in sample and ("messages" in sample or "prompt" in sample)


def to_rl(sample: dict[str, Any], images: list[str] | None = None) -> dict[str, Any]:
    """An RL sample as {"messages", "answer"}. ValueError says why it cannot be used. `images`: as in `to_messages`."""
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
    row = {"messages": _prompt_of(chat, images), "answer": answer}
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


def _prompt_of(chat: dict[str, Any], images: list[str] | None = None) -> list[Message]:
    """An RL or preference sample's prompt: the chat without its last message, the reply to its last user message."""
    messages = to_messages(chat, images)
    if not messages[-2]["content"].strip():
        raise ValueError("needs a non-empty last user message: the prompt the model replies to")
    return messages[:-1]


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
    messages = _prompt_of({**sample, "messages": prompt + [chosen]})  # the prompt follows the same rules as chat samples
    return {"messages": messages, "chosen": chosen, "rejected": rejected}


def to_sample(sample: Any, images: list[str] | None = None) -> dict[str, Any]:
    """The sample as the server stores it: {"messages"} for a chat, {"messages", "answer"} for RL, {"messages",
    "chosen", "rejected"} for DPO. `images`: as in `to_messages`, for a chat or RL."""
    if is_rl(sample):
        return to_rl(sample, images)
    if is_preference(sample):
        return to_preference(sample)
    if is_tool_sample(sample):
        if any(_media_parts(sample)):
            raise ValueError("has tool calls; samples with tools and images or videos are not taken yet")
        return to_tool_chat(sample)
    return {"messages": to_messages(sample, images)}


def _line(sample: Any) -> tuple[Any, str | None]:
    """The sample as a line of a JSONL file: in the server's form, or as it is when it has images or videos (they stay
    content parts). Second, for a sample without them whose text holds a media tag, the error it is in a file with them."""
    images: list[str] = []
    try:
        row = to_sample(sample, images)
    except _Tagged as error:
        if any(_media_parts(sample)):
            raise
        return to_sample(sample), str(error)
    return (sample if images else row), None


def write_jsonl(samples: Iterable[Any], path: str | os.PathLike[str]) -> int:
    """Write samples in any format above as JSONL in the server's form; returns how many. Raises on the first bad one.
    A sample with images or videos is written as it is; they are read, paths relative to `path`'s folder."""
    count = 0
    with_images = False
    tagged = None   # the first sample without images or videos whose text holds a media tag
    with Path(path).open("w", encoding="utf-8") as out:
        for number, sample in enumerate(samples, 1):
            try:
                row, tag_error = _line(sample)
                for _, _, kind, source in _media_parts(row):
                    read_media(kind, source, Path(path).parent)
                    with_images = True
            except ValueError as error:
                raise ValueError(f"sample {number}: {error}") from None
            if tag_error and tagged is None:
                tagged = f"sample {number}: {tag_error}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if with_images and tagged:
        raise ValueError(tagged)
    return count


@dataclass
class CheckResult:
    samples: int = 0
    chars: int = 0  # message text; a token is roughly 3-4 characters of English
    errors: list[str] = field(default_factory=list)
    kind: str | None = None  # "chat", "rl" for prompts with answers, or "preference" for DPO pairs
    images: int = 0  # image parts in the samples; with any, or any video, the upload is a bundle (write_bundle)
    videos: int = 0  # video parts in the samples

    @property
    def ok(self) -> bool:
        return not self.errors


def check_file(path: str | os.PathLike[str]) -> CheckResult:
    """Run the server's format check on a local JSONL file. Token counts per model come from the server after upload."""
    path = Path(path)
    result = CheckResult()
    size = path.stat().st_size
    if size > BUNDLE_MAX_BYTES:
        result.errors.append(f"the file is over {BUNDLE_MAX_BYTES // 1000**3} GB")
        return result
    unique: dict[str, int] = {}   # SHA-256 -> bytes of each image or video, packed once in a bundle
    references = 0   # characters of the samples that name images and videos; a bundle names each by its SHA-256
    kinds = set()
    tagged: list[str] = []   # lines without images or videos whose text holds a media tag: errors if the file has any
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
                sample = json.loads(line)
                try:
                    row = to_sample(sample, found)
                except _Tagged as error:
                    if any(_media_parts(sample)):
                        raise
                    row = to_sample(sample)
                    if len(tagged) < MAX_ERRORS:
                        tagged.append(f"line {number}: {error}")
                videos = 0
                for _, _, kind, source in _media_parts(sample):
                    _, data = read_media(kind, source, path.parent)
                    unique[hashlib.sha256(data).hexdigest()] = len(data)
                    references += len(source) - IMAGE_REF_CHARS
                    videos += kind == "video"
            except json.JSONDecodeError as error:
                _note(result, f"line {number}: not valid JSON ({error.msg})")
                continue
            except ValueError as error:
                _note(result, f"line {number}: {error}")
                continue
            result.samples += 1
            result.images += len(found) - videos
            result.videos += videos
            replies = [row[k] for k in ("chosen", "rejected") if k in row]
            result.chars += sum(len(m["content"]) for m in row["messages"] + replies)
            kinds.add("rl" if "answer" in row else "preference" if "chosen" in row else "chat")
    if result.images or result.videos:   # the upload is a bundle
        result.errors = sorted(result.errors + tagged, key=lambda error: int(error.split(":")[0].removeprefix("line ")))[:MAX_ERRORS]
        samples_bytes = size - references
        if len(unique) > BUNDLE_MAX_IMAGES:
            result.errors.append(f"the file has {len(unique):,} different images and videos; a dataset takes at most "
                                 f"{BUNDLE_MAX_IMAGES:,}")
        if samples_bytes > MAX_FILE_BYTES:
            result.errors.append(f"the samples without their images and videos are over {MAX_FILE_BYTES // 1000**3} GB")
        elif samples_bytes + sum(unique.values()) > BUNDLE_MAX_BYTES:
            result.errors.append(f"the samples and their images and videos are over {BUNDLE_MAX_BYTES // 1000**3} GB")
    elif size > MAX_FILE_BYTES:
        result.errors.append(f"the file is over {MAX_FILE_BYTES // 1000**3} GB")
    if not result.errors and not result.samples:
        result.errors.append("the file has no samples")
    if len(kinds) > 1:
        result.errors.append("the file mixes chat samples, prompts with answers (RL) and preference pairs; a dataset holds one kind")
    result.kind = kinds.pop() if len(kinds) == 1 else None
    return result


def write_bundle(path: str | os.PathLike[str], out: str | os.PathLike[str]) -> int:
    """Pack a JSONL file whose samples carry images or videos into the tar the server takes: each file once, as
    images/<SHA-256 of its bytes>.jpg|png or videos/<SHA-256>.mp4, then the samples as data.jsonl with every image and
    video named "sha256:...". Returns the number of files. ValueError names the first line that cannot be packed."""
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
                for parts, i, kind, source in _media_parts(sample):
                    try:
                        extension, data = read_media(kind, source, path.parent)
                    except ValueError as error:
                        raise ValueError(f"line {number}: {error}") from None
                    sha = hashlib.sha256(data).hexdigest()
                    if sha not in packed:
                        packed.add(sha)
                        add(f"{kind}s/{sha}.{extension}", len(data), io.BytesIO(data))
                    parts[i] = {"type": kind, kind: f"sha256:{sha}"}
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
    trained on are left out and counted. Samples with images or videos keep them, their paths rewritten for `out_dir`. Shuffled
    with `seed`; the validation set is `val_fraction` of the rows, at most 5,000."""
    import random

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be at least 0 and below 1")
    rows = read_rows(path)
    prompt_columns = [prompt_column] if isinstance(prompt_column, str) else prompt_column or []
    if response_column and answer_column:
        raise ValueError("give response_column (chat samples) or answer_column (RL samples), not both")
    target_column = response_column or answer_column
    if bool(prompt_columns) != bool(target_column):
        raise ValueError("prompt_column goes with response_column or answer_column: give both, or neither for rows "
                         "already in an accepted format")
    if prompt_columns and target_column and rows and isinstance(rows[0], dict):
        missing = [c for c in (*prompt_columns, target_column) if c not in rows[0]]
        if missing:
            raise ValueError(f"{Path(path).name} has no column {', '.join(missing)}; it has {', '.join(rows[0])}")
    kept: list[tuple[int, Any, str | None]] = []   # row number, line, its error in a file with images or videos
    with_images = False
    errors: list[str] = []
    skipped = 0
    out = Path(out_dir)
    for number, row in enumerate(rows, 1):
        try:
            if prompt_columns and target_column:
                missing = [c for c in (*prompt_columns, target_column) if not isinstance(row, dict) or c not in row]
                if missing:   # an empty value is the row's choice, a missing column is a broken row
                    raise ValueError(f"has no column {', '.join(missing)}")
                prompt = "\n\n".join(str(row[c]) for c in prompt_columns if row[c])
                row = {"prompt": prompt, "completion" if response_column else "answer": row[target_column]}
            if system and isinstance(row, dict):
                row = dict(row, system=system)
            line, tag_error = _line(row)
            for parts, i, kind, source in _media_parts(line):
                read_media(kind, source, Path(path).parent)
                if not source.startswith("data:"):   # a path: train.jsonl may be in another folder
                    # resolved: ".." from a symlinked folder (/tmp on macOS) is not its parent in the path
                    file = os.path.relpath((Path(path).parent / source).resolve(), out.resolve())
                    parts[i] = {"type": kind, kind: file}
            kept.append((number, line, tag_error))
            if any(_media_parts(line)):
                with_images = True
        except ValueError as error:
            skipped += 1
            if len(errors) < MAX_ERRORS:
                errors.append(f"row {number}: {error}")
    samples: list[Any] = []
    for number, line, tag_error in kept:
        if with_images and tag_error:
            skipped += 1
            if len(errors) < MAX_ERRORS:
                errors.append(f"row {number}: {tag_error}")
        else:
            samples.append(line)
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
