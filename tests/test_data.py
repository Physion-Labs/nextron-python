import json
import struct
from pathlib import Path

import pytest

from nextron_cloud import check_file, to_messages, write_jsonl
from nextron_cloud.data import prepare_file, to_sample

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
    ({"messages": [USER, ASSISTANT], "images": ["a.png"]}, "images go in the user messages"),
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
    rows = [{"question": "1+1?", "context": "sums", "reply": "2"}, {"question": "2+2?", "reply": "4"}, {"question": "3+3?"}]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = prepare_file(path, tmp_path / "out", val_fraction=0, prompt_column=["question", "context"], response_column="reply")
    assert (report["train"]["samples"], report["skipped"]) == (2, 1)


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
    assert write_jsonl([tagged], tmp_path / "out.jsonl") == 1
    with pytest.raises(ValueError, match="sample 1: has <image>"):
        write_jsonl([both], tmp_path / "out.jsonl")
