import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys

import pytest


MODULE = Path(__file__).resolve().parents[1] / "funasr/metrics/compute_eer.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("eer_boundary_helper", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def files(tmp_path, order=(0, 1, 2, 3), separator=" "):
    trials, scores = tmp_path / "trials.txt", tmp_path / "scores.txt"
    pairs = [("a", "b", "target", "0.9"), ("a", "c", "target", "0.8"),
             ("b", "a", "nontarget", "0.2"), ("c", "a", "nontarget", "0.1")]
    trials.write_text("\n".join(separator.join(row[:3]) for row in pairs) + "\n")
    scores.write_text("\n".join(separator.join((pairs[i][0], pairs[i][1], pairs[i][3]))
                                  for i in order) + "\n")
    return trials, scores


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_eer_and_threshold_are_invariant_to_all_score_orders(tmp_path, order):
    trials, scores = files(tmp_path, order)
    assert load_helper().compute_eer(trials, scores) == (0.0, 0.8)


@pytest.mark.parametrize("separator", [" ", "  ", "\t"])
def test_eer_accepts_whitespace_separated_pair_keys(tmp_path, separator):
    trials, scores = files(tmp_path, (3, 2, 1, 0), separator)
    assert load_helper().compute_eer(str(trials), str(scores)) == (0.0, 0.8)


def test_eer_subset_uses_only_scored_trials(tmp_path):
    trials, scores = files(tmp_path, (3, 0))
    assert load_helper().compute_eer(trials, scores) == (0.0, 0.9)


def test_eer_rejects_unknown_pair_with_path_objects(tmp_path):
    trials, scores = files(tmp_path)
    scores.write_text("b c 0.9\n")
    with pytest.raises(Exception) as error:
        load_helper().compute_eer(trials, scores)
    assert str(error.value) == f"Missing entry for b and c {scores}"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_eer_rejects_nonfinite_scores(tmp_path, value):
    trials, scores = files(tmp_path)
    scores.write_text(f"a b {value}\nc a 0.1\n")
    with pytest.raises(ValueError):
        load_helper().compute_eer(trials, scores)


def test_eer_cli_outputs_percent_and_threshold(tmp_path):
    trials, scores = files(tmp_path, (3, 2, 1, 0))
    result = subprocess.run([sys.executable, str(MODULE), str(trials), str(scores)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "EER is 0.0000 at threshold 0.8000"


def test_eer_cli_returns_nonzero_for_unknown_pair(tmp_path):
    trials, scores = files(tmp_path)
    scores.write_text("b c 0.9\n")
    result = subprocess.run([sys.executable, str(MODULE), str(trials), str(scores)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "EER is" not in result.stdout
    assert f"Missing entry for b and c {scores}" in result.stderr
