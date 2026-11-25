import json
import struct
from pathlib import Path

import pytest

from nextron_cloud import check_file, to_messages, write_jsonl
from nextron_cloud.data import prepare_file, to_sample
from tests.conftest import mp4

USER, ASSISTANT = {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}


@pytest.mark.parametrize("sample", [
    {"messages": [USER, ASSISTANT]},
    {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]},
    {"instruction": "q", "output": "a"},
    {"prompt": "q", "completion": "a"},
    {"question": "q", "answer": "a"},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}, ASSISTANT]},
    {"instruction": "q", "response": "a", "category": "open_qa"},
    {"conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    {"problem": "q", "solution": "a"},
    {"input": "q", "output": "a"},
    {"question": "q", "response": "a"},
])
def test_known_formats_become_messages(sample):
    assert to_messages(sample) == [USER, ASSISTANT]


@pytest.mark.parametrize("sample, error", [
    ({"messages": [USER]}, "assistant"),
    ({"messages": [USER, ASSISTANT], "images": ["a.png"]}, "images and videos go in the user messages"),
    ({"messages": [USER, {"role": "system", "content": "s"}, ASSISTANT]}, "first"),
])
def test_unusable_samples_say_why(sample, error):
    with pytest.raises(ValueError, match=error):
        to_messages(sample)


def test_check_file_reports_line_numbers(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"prompt": "q", "completion": "a"}).encode() + b"\n\n{oops\n")
    result = check_file(path)
    assert result.samples == 1
    assert result.errors == ["line 3: not valid JSON (Expecting property name enclosed in double quotes)"]


def test_write_jsonl_output_passes_the_check(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    assert write_jsonl([{"query": "q", "response": "a"}, {"instruction": "q", "output": "a"}], path) == 2
    assert check_file(path).ok


def test_prepare_file_splits_a_csv_by_column(tmp_path: Path):
    from nextron_cloud.data import prepare_file
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("ticket,reply\n" + "".join(f"q{i},a{i}\n" for i in range(19)) + "q19,\n")
    report = prepare_file(csv_path, tmp_path, prompt_column="ticket", response_column="reply", system="Be kind.")
    assert (report["train"]["samples"], report["val"]["samples"], report["skipped"]) == (17, 2, 1)
    assert check_file(tmp_path / "train.jsonl").ok
    first = json.loads((tmp_path / "val.jsonl").read_text().splitlines()[0])["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]


def test_prepare_file_names_a_missing_column(tmp_path: Path):
    from nextron_cloud.data import prepare_file
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("ticket,answer\nq,a\n")
    with pytest.raises(ValueError, match="no column reply; it has ticket, answer"):
        prepare_file(csv_path, tmp_path, prompt_column="ticket", response_column="reply")


def test_prepare_file_refuses_column_options_that_do_not_go_together(tmp_path: Path):
    from nextron_cloud.data import prepare_file
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("ticket,reply\nq,a\n")
    for columns, error in [({"prompt_column": "ticket"}, "give both"), ({"response_column": "reply"}, "give both"),
                           ({"prompt_column": "ticket", "response_column": "reply", "answer_column": "reply"}, "not both")]:
        with pytest.raises(ValueError, match=error):
            prepare_file(csv_path, tmp_path, **columns)


def test_prepare_file_leaves_out_a_row_that_lacks_a_column(tmp_path: Path):
    from nextron_cloud.data import prepare_file
    rows = [{"question": "1+1?", "context": "sums", "reply": "2"}, {"question": "2+2?", "context": "", "reply": "4"},
            {"question": "3+3?", "reply": "6"}, {"question": "4+4?", "context": "sums"}, None, ["5+5?", "sums", "10"]]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    report = prepare_file(path, tmp_path / "out", val_fraction=0, prompt_column=["question", "context"], response_column="reply")
    assert (report["train"]["samples"], report["skipped"]) == (2, 4)   # an empty context is kept, a missing one is not
    assert report["errors"][:2] == ["row 3: has no column context", "row 4: has no column reply"]
    assert prepare_file(path, tmp_path / "plain", val_fraction=0)["skipped"] == 6   # no columns named: none is a sample


def test_preference_pairs_in_three_formats(tmp_path: Path):
    q = [{"role": "user", "content": "say hi"}]
    hi, no = {"role": "assistant", "content": "hi"}, {"role": "assistant", "content": "no"}
    want = {"messages": q, "chosen": hi, "rejected": no}
    assert to_sample({"messages": q, "chosen": "hi", "rejected": no}) == want
    assert to_sample({"prompt": "say hi", "chosen": "hi", "rejected": "no"}) == want
    assert to_sample({"chosen": q + [hi], "rejected": q + [no]}) == want
    for bad in ({"prompt": "q", "chosen": "a", "rejected": ""}, {"chosen": q + [hi], "rejected": [no]}):
        with pytest.raises(ValueError):
            to_sample(bad)
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n")
    assert (check_file(path).ok, check_file(path).kind) == (True, "preference")


def test_rl_samples_keep_their_answer_and_a_file_holds_one_kind(tmp_path: Path):
    assert to_sample({"prompt": "2+2?", "answer": 4}) == {"messages": [{"role": "user", "content": "2+2?"}], "answer": "4"}
    assert to_sample({"question": "2+2?", "answer": "4"})["messages"][-1] == {"role": "assistant", "content": "4"}  # a chat pair
    for bad in ({"prompt": "q", "answer": ""}, {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "answer": "a"}):
        with pytest.raises(ValueError):
            to_sample(bad)
    path = tmp_path / "rl.jsonl"
    path.write_text(json.dumps({"prompt": "q", "answer": "a"}) + "\n")
    assert (check_file(path).ok, check_file(path).kind) == (True, "rl")
    path.write_text(json.dumps({"prompt": "q", "answer": "a"}) + "\n" + json.dumps({"prompt": "q", "completion": "a"}) + "\n")
    assert "mixes" in check_file(path).errors[0]
    csv = tmp_path / "math.csv"
    csv.write_text("question,final\nwhat is 2+2,4\nwhat is 3+3,6\n")
    report = prepare_file(csv, tmp_path / "out", val_fraction=0, prompt_column="question", answer_column="final")
    answers = sorted(json.loads(line)["answer"] for line in Path(report["train"]["path"]).read_text().splitlines())
    assert answers == ["4", "6"]


def test_samples_with_images_are_checked_and_packed(tmp_path: Path):
    import hashlib
    import tarfile

    from nextron_cloud.data import write_bundle

    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 640, 480)
    jpg = b"\xff\xd8\xff\xc0" + struct.pack(">HBHH", 8, 8, 480, 640) + b"\x03\xff\xda"
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "cat.png").write_bytes(png)
    uri = "data:image/jpeg;base64," + __import__("base64").b64encode(jpg).decode()
    rows = [
        {"messages": [{"role": "user", "content": [{"type": "image", "image": "photos/cat.png"}, {"type": "text", "text": "what"}]},
                      {"role": "assistant", "content": "a cat"}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                   {"type": "image", "image": "photos/cat.png"}, {"type": "text", "text": "both"}]},
                      {"role": "assistant", "content": "two"}]},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = check_file(path)
    assert result.ok and (result.samples, result.images) == (2, 3)
    assert write_bundle(path, tmp_path / "bundle.tar") == 2   # the same file twice is packed once
    a, b = hashlib.sha256(png).hexdigest(), hashlib.sha256(jpg).hexdigest()
    with tarfile.open(tmp_path / "bundle.tar") as tar:
        assert tar.getnames() == [f"images/{a}.png", f"images/{b}.jpg", "data.jsonl"]   # the samples last
        packed = [json.loads(line) for line in tar.extractfile("data.jsonl")]
    assert [p["image"] for p in packed[1]["messages"][0]["content"][:2]] == [f"sha256:{b}", f"sha256:{a}"]

    for image, error in [("https://example.com/cat.png", "download it"), ("photos/missing.png", "cannot be read"),
                         ("data.jsonl", "neither JPEG nor PNG")]:
        path.write_text(json.dumps({"messages": [{"role": "user", "content": [{"type": "image", "image": image}]},
                                                 {"role": "assistant", "content": "x"}]}) + "\n")
        assert error in check_file(path).errors[0]

    for name, data, error in [("cut.png", png[:20], "not a complete PNG"), ("cut.jpg", jpg[:4], "not a complete JPEG"),
                              ("wide.png", png[:16] + struct.pack(">II", 9000, 9000), "36 million pixels"),
                              ("strip.png", png[:16] + struct.pack(">II", 8000, 32), "200 times longer")]:
        (tmp_path / name).write_bytes(data)
        path.write_text(json.dumps({"messages": [{"role": "user", "content": [{"type": "image", "image": name}]},
                                                 {"role": "assistant", "content": "x"}]}) + "\n")
        assert error in check_file(path).errors[0]


def test_samples_with_videos_are_checked_and_packed(tmp_path: Path):
    import hashlib
    import tarfile

    from nextron_cloud.data import write_bundle

    video = mp4(64, 48, 2)
    (tmp_path / "race.mp4").write_bytes(video)
    (tmp_path / "short.mp4").write_bytes(mp4(64, 48, 0.5))
    sha = hashlib.sha256(video).hexdigest()
    path = tmp_path / "data.jsonl"

    def write(*videos):
        content = [{"type": "video", "video": v} for v in videos] + [{"type": "text", "text": "who wins?"}]
        path.write_text(json.dumps({"messages": [{"role": "user", "content": content}, ASSISTANT]}) + "\n")

    write("race.mp4")
    result = check_file(path)
    assert result.ok and (result.images, result.videos) == (0, 1)
    assert write_bundle(path, tmp_path / "bundle.tar") == 1
    with tarfile.open(tmp_path / "bundle.tar") as tar:
        assert tar.getnames() == [f"videos/{sha}.mp4", "data.jsonl"]
        packed = json.loads(tar.extractfile("data.jsonl").read())
    assert packed["messages"][0]["content"][0] == {"type": "video", "video": f"sha256:{sha}"}
    report = prepare_file(path, tmp_path / "out", val_fraction=0)   # the video's path rewritten for the new folder
    assert check_file(report["train"]["path"]).videos == 1

    for videos, error in [(["data.jsonl"], "is not an MP4 file"), (["short.mp4"], "is shorter than a second"),
                          (["https://example.com/race.mp4"], "download it"), (["data:video/mp4;base64,AAAA"], "data: URI"),
                          (["race.mp4"] * 5, "has 5 videos")]:
        write(*videos)
        assert error in check_file(path).errors[0]


def test_rl_prompts_with_images_are_checked(tmp_path: Path):
    (tmp_path / "cat.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 640, 480))
    row = {"messages": [{"role": "user", "content": [{"type": "image", "image": "cat.png"}, {"type": "text", "text": "what"}]}],
           "answer": "cat"}
    (tmp_path / "rl.jsonl").write_text(json.dumps(row) + "\n")
    result = check_file(tmp_path / "rl.jsonl")
    assert result.ok and result.kind == "rl" and result.images == 1


def test_prepare_and_write_jsonl_keep_images(tmp_path: Path):
    from nextron_cloud.data import write_jsonl

    (tmp_path / "cat.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 640, 480))
    row = {"messages": [{"role": "user", "content": [{"type": "image", "image": "cat.png"}, {"type": "text", "text": "what"}]},
                        {"role": "assistant", "content": "a cat"}]}
    assert write_jsonl([row], tmp_path / "data.jsonl") == 1
    assert json.loads((tmp_path / "data.jsonl").read_text()) == row
    report = prepare_file(tmp_path / "data.jsonl", tmp_path / "out", val_fraction=0)
    assert report["skipped"] == 0
    result = check_file(report["train"]["path"])   # the image is found from the new folder
    assert result.ok and result.images == 1
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "a" / "b")   # like /tmp on macOS: ".." from it is not `tmp_path`
    report = prepare_file(tmp_path / "data.jsonl", tmp_path / "link", val_fraction=0)
    assert check_file(report["train"]["path"]).ok


def test_a_media_tag_in_text_is_refused_only_in_a_file_with_images(tmp_path: Path):
    (tmp_path / "cat.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 640, 480))
    tagged = {"messages": [{"role": "user", "content": "What does <image> mean in HTML?"}, {"role": "assistant", "content": "Nothing."}]}
    picture = {"messages": [{"role": "user", "content": [{"type": "image", "image": "cat.png"}, {"type": "text", "text": "what"}]},
                            {"role": "assistant", "content": "a cat"}]}
    both = {"messages": [{"role": "user", "content": [{"type": "image", "image": "cat.png"}, {"type": "text", "text": "an <image> tag"}]},
                         {"role": "assistant", "content": "a cat"}]}
    path = tmp_path / "data.jsonl"

    def check(*rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return check_file(path)

    assert check(tagged).ok and check(tagged).samples == 1               # as the server takes a file without images
    assert [e[:7] for e in check(picture, tagged).errors] == ["line 2:"]   # a bundle: the server refuses the tag
    assert [e[:7] for e in check(tagged, {"oops": 1}, picture).errors] == ["line 1:", "line 2:"]
    assert [e[:7] for e in check(both).errors] == ["line 1:"]
    for row in ({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "an <image> tag"}]},
                {"system": "<audio> is a tag", "messages": [USER, ASSISTANT]},
                {"prompt": "q", "completion": "<video>"}):
        assert [e[:7] for e in check(picture, row).errors] == ["line 2:"]   # every role and format of a file with images
    assert write_jsonl([tagged], tmp_path / "out.jsonl") == 1
    with pytest.raises(ValueError, match="sample 1: has <image>"):
        write_jsonl([both], tmp_path / "out.jsonl")
    with pytest.raises(ValueError, match="sample 2: has <image>"):
        write_jsonl([picture, tagged], tmp_path / "out.jsonl")
    with pytest.raises(ValueError, match="sample 1: .*cannot be read"):
        write_jsonl([{"messages": [{"role": "user", "content": [{"type": "image", "image": "missing.png"}]}, ASSISTANT]}],
                    tmp_path / "out.jsonl")

    missing = {"messages": [{"role": "user", "content": [{"type": "image", "image": "missing.png"}]}, ASSISTANT]}
    path.write_text("".join(json.dumps(r) + "\n" for r in (picture, tagged, missing)))
    report = prepare_file(path, tmp_path / "out", val_fraction=0)
    assert report["skipped"] == 2 and [e[:6] for e in sorted(report["errors"])] == ["row 2:", "row 3:"]
    assert check_file(report["train"]["path"]).ok


def test_a_file_with_images_is_held_to_the_bundle_limits(tmp_path: Path, monkeypatch):
    from nextron_cloud import data

    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 64, 48) + bytes(600)
    uri = "data:image/png;base64," + __import__("base64").b64encode(png).decode()
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": [{"type": "image", "image": uri}]}, ASSISTANT]}) + "\n")
    monkeypatch.setattr(data, "MAX_FILE_BYTES", 500)
    assert check_file(path).ok   # over the limit of a file, but its samples without the image are not
    (tmp_path / "big.png").write_bytes(png)
    path.write_text(json.dumps({"messages": [{"role": "user", "content": [{"type": "image", "image": "big.png"}]}, ASSISTANT]}) + "\n")
    monkeypatch.setattr(data, "BUNDLE_MAX_BYTES", 600)
    assert "the samples and their images and videos are over" in check_file(path).errors[0]
    monkeypatch.setattr(data, "BUNDLE_MAX_IMAGES", 0)
    assert "different images" in check_file(path).errors[0]
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "x" * 600}, ASSISTANT]}) + "\n")
    assert check_file(path).errors == ["the file is over 0 GB"]


def test_an_rl_or_preference_sample_with_a_blank_prompt_is_refused():
    for sample in ({"prompt": "   ", "answer": "4"}, {"messages": [{"role": "user", "content": "\n"}], "chosen": "a", "rejected": "b"}):
        with pytest.raises(ValueError, match="needs a non-empty last user message"):
            to_sample(sample)


WEATHER = {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
CALL = {"messages": [{"role": "user", "content": "Weather in Paris?"},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": "18C"}, {"role": "assistant", "content": "It is 18C."}],
        "tools": [{"type": "function", "function": WEATHER}]}
DOLCI = {"messages": [{"role": "system", "content": "SYS", "functions": json.dumps([WEATHER])}, {"role": "user", "content": "Weather in Paris?"},
                      {"role": "assistant", "content": "", "function_calls": 'get_weather(city="Paris")'},
                      {"role": "environment", "content": "18C"}, {"role": "assistant", "content": "It is 18C."}]}


@pytest.mark.parametrize("sample", [CALL, DOLCI])
def test_tool_samples_become_openai_messages_with_tools(sample):
    row = to_sample(sample)
    call = row["messages"][-3]["tool_calls"][0]
    assert call["function"] == {"name": "get_weather", "arguments": {"city": "Paris"}} and row["messages"][-2]["role"] == "tool"
    assert row["tools"] == [{"type": "function", "function": WEATHER}] and ("ai2" in row) == (sample is DOLCI)


def test_rl_prompts_and_preference_pairs_take_tools():
    prompt, call = CALL["messages"][:1], to_sample(CALL)["messages"][1]
    rl = to_sample({"tools": CALL["tools"], "messages": CALL["messages"][:3], "answer": 18})   # ends with the tool's result
    assert rl == {"messages": to_sample(CALL)["messages"][:3], "tools": CALL["tools"], "answer": "18"}
    pair = to_sample({"tools": CALL["tools"], "messages": prompt, "chosen": CALL["messages"][1], "rejected": "I cannot look that up."})
    assert pair == {"chosen": call, "rejected": {"role": "assistant", "content": "I cannot look that up."},
                    "messages": prompt, "tools": CALL["tools"]}


def test_check_file_refuses_a_call_to_an_unknown_tool(tmp_path: Path):
    bad = {**CALL, "tools": [{"type": "function", "function": {"name": "get_time"}}]}
    path = tmp_path / "tools.jsonl"
    path.write_text(json.dumps(CALL) + "\n" + json.dumps(bad) + "\n")
    result = check_file(path)
    assert result.samples == 1 and result.errors == ["line 2: calls 'get_weather', which is not in the sample's tools"]
