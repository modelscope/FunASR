import importlib.util
from pathlib import Path

import pytest


def _load_helper():
    path = Path(__file__).resolve().parent.parent / "funasr/metrics/compute_eer.py"
    spec = importlib.util.spec_from_file_location("metrics_compute_eer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (2, 3, 0, 1), (3, 0)])
def test_compute_eer_pairs_scores_with_trials_by_key(tmp_path, order):
    helper = _load_helper()
    trials_file = tmp_path / "trials.txt"
    scores_file = tmp_path / "scores.txt"
    trials_file.write_text(
        "utt1 utt2 target\nutt1 utt3 target\n"
        "utt4 utt2 nontarget\nutt4 utt3 nontarget\n",
        encoding="utf-8",
    )
    scores = ["utt1 utt2 0.9\n", "utt1 utt3 0.8\n", "utt4 utt2 0.2\n", "utt4 utt3 0.1\n"]
    scores_file.write_text("".join(scores[i] for i in order), encoding="utf-8")

    eer, threshold = helper.compute_eer(str(trials_file), str(scores_file))

    assert eer == 0.0


def test_compute_eer_raises_for_a_scored_trial_missing_from_trials(tmp_path):
    helper = _load_helper()
    trials_file = tmp_path / "trials.txt"
    scores_file = tmp_path / "scores.txt"
    trials_file.write_text("utt1 utt2 target\n", encoding="utf-8")
    scores_file.write_text("utt1 utt3 0.9\n", encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        helper.compute_eer(str(trials_file), str(scores_file))

    assert str(exc_info.value) == "Missing entry for utt1 and utt3 " + str(scores_file)
