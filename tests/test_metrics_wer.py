import importlib.util
from pathlib import Path

import pytest

from funasr.metrics.wer import compute_wer, compute_wer_by_line, print_cer_detail

# Standalone copies of compute_wer_by_line kept for the runtime bundles and the
# Aishell examples. They duplicate the helper instead of importing it, so they
# need the same boundary coverage as the library version.
DUPLICATE_WER_HELPERS = [
    "runtime/python/utils/compute_wer.py",
    "runtime/python/libtorch/funasr_torch/utils/compute_wer.py",
    "examples/aishell/paraformer/utils/compute_wer.py",
    "examples/aishell/e_paraformer/utils/compute_wer.py",
]


def _load_helper(relative_path):
    path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(
        f"dup_{relative_path.replace('/', '_').removesuffix('.py')}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative_path", DUPLICATE_WER_HELPERS)
def test_duplicate_wer_helpers_handle_inputs_at_the_int16_boundary(relative_path):
    helper = _load_helper(relative_path)

    out_item = helper.compute_wer_by_line(["a"] * 32768, ["a"])

    assert out_item["nwords"] == 1
    assert out_item["cor"] == 1
    assert out_item["wrong"] == 32767
    assert out_item["ins"] == 32767


@pytest.mark.parametrize("relative_path", DUPLICATE_WER_HELPERS)
def test_duplicate_wer_helpers_agree_with_the_library(relative_path):
    helper = _load_helper(relative_path)

    for hyp, ref in [
        (["a", "b", "c"], ["a", "x", "c"]),
        (["uh"], []),
        (["the", "cat", "sat"], ["the", "cat", "sat", "down"]),
    ]:
        assert {
            k: int(v) for k, v in helper.compute_wer_by_line(hyp, ref).items()
        } == {k: int(v) for k, v in compute_wer_by_line(hyp, ref).items()}


def test_cer_detail_for_empty_reference_does_not_divide_by_zero():
    out_item = compute_wer_by_line(["uh"], [])

    assert out_item["nwords"] == 0
    assert out_item["ins"] == 1
    assert print_cer_detail(out_item).endswith("corr:0.00%,cer:100.00%")


def test_wer_by_line_handles_inputs_at_the_int16_boundary():
    # 32768 tokens used to raise OverflowError: the cost matrix was int16, so
    # initialising its first row/column could not store the index 32768.
    out_item = compute_wer_by_line(["a"] * 32768, ["a"])

    assert out_item["nwords"] == 1
    assert out_item["cor"] == 1
    assert out_item["wrong"] == 32767
    assert out_item["ins"] == 32767


def test_wer_by_line_handles_a_long_reference_at_the_int16_boundary():
    out_item = compute_wer_by_line(["a"], ["a"] * 32768)

    assert out_item["nwords"] == 32768
    assert out_item["cor"] == 1
    assert out_item["wrong"] == 32767
    assert out_item["del"] == 32767


def test_compute_wer_scores_all_utterances_when_one_reference_is_empty(tmp_path):
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    cer_file = tmp_path / "out.wer"
    ref_file.write_text("utt1 the cat sat\nutt2\nutt3 good morning\n", encoding="utf-8")
    hyp_file.write_text("utt1 the cat sat\nutt2 uh\nutt3 good morning\n", encoding="utf-8")

    compute_wer(str(ref_file), str(hyp_file), str(cer_file))

    report = cer_file.read_text(encoding="utf-8")
    assert "utt2(nwords=0,cor=0,ins=1,del=0,sub=0) corr:0.00%,cer:100.00%" in report
    assert "utt3(nwords=2,cor=2,ins=0,del=0,sub=0) corr:100.00%,cer:0.00%" in report
    assert "%WER 20.0 [ 1 / 5, 1 ins, 0 del, 0 sub ]" in report
    assert "%SER 33.33 [ 1 / 3 ]" in report


def _write_summary_fixture(tmp_path):
    # utt2 has a reference but no hypothesis; utt9 has a hypothesis but no
    # reference. Neither can be scored.
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1 the cat sat\nutt2 hello world\nutt3 good morning\n", encoding="utf-8")
    hyp_file.write_text(
        "utt1 the cat sat\nutt3 good morning\nutt9 stray output\n", encoding="utf-8"
    )
    return ref_file, hyp_file, tmp_path / "out.wer"


def test_compute_wer_summary_counts_references_missing_from_hyp(tmp_path):
    ref_file, hyp_file, cer_file = _write_summary_fixture(tmp_path)

    compute_wer(str(ref_file), str(hyp_file), str(cer_file))

    report = cer_file.read_text(encoding="utf-8")
    assert "%SER 0.0 [ 0 / 2 ]" in report
    assert "Scored 2 sentences, 1 not present in hyp." in report


@pytest.mark.parametrize("relative_path", DUPLICATE_WER_HELPERS)
def test_duplicate_wer_helpers_count_references_missing_from_hyp(relative_path, tmp_path):
    helper = _load_helper(relative_path)
    ref_file, hyp_file, cer_file = _write_summary_fixture(tmp_path)

    helper.compute_wer(str(ref_file), str(hyp_file), str(cer_file))

    report = cer_file.read_text(encoding="utf-8")
    assert "Scored 2 sentences, 1 not present in hyp." in report
