from funasr.metrics.wer import compute_wer, compute_wer_by_line, print_cer_detail


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
