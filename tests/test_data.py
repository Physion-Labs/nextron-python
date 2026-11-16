import json
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
    ({"messages": [USER, ASSISTANT], "images": ["a.png"]}, "text only"),
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
