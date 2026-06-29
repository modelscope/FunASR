import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from funasr.models.qwen3_asr import model as qwen3_model


def test_qwen3_asr_dependency_check_reports_transformers_mismatch(monkeypatch):
    def fake_version(package_name):
        versions = {
            "qwen-asr": "0.0.6",
            "transformers": "5.5.0",
        }
        return versions[package_name]

    def fake_requires(package_name):
        assert package_name == "qwen-asr"
        return ["transformers==4.57.6", "accelerate==1.12.0"]

    monkeypatch.setattr(qwen3_model, "_package_version", fake_version)
    monkeypatch.setattr(qwen3_model, "_package_requires", fake_requires)

    with pytest.raises(ImportError) as exc_info:
        qwen3_model._check_qwen3_asr_dependencies()

    message = str(exc_info.value)
    assert "Qwen3-ASR dependency mismatch" in message
    assert "qwen-asr==0.0.6 requires transformers==4.57.6" in message
    assert "active environment has transformers==5.5.0" in message
    assert 'pip install -U "qwen-asr==0.0.6" "transformers==4.57.6" accelerate' in message


def test_qwen3_asr_dependency_check_accepts_matching_transformers(monkeypatch):
    def fake_version(package_name):
        versions = {
            "qwen-asr": "0.0.6",
            "transformers": "4.57.6",
        }
        return versions[package_name]

    def fake_requires(package_name):
        assert package_name == "qwen-asr"
        return ["transformers==4.57.6", "accelerate==1.12.0"]

    monkeypatch.setattr(qwen3_model, "_package_version", fake_version)
    monkeypatch.setattr(qwen3_model, "_package_requires", fake_requires)

    qwen3_model._check_qwen3_asr_dependencies()


def test_qwen3_asr_dependency_check_matches_requirement_names_case_insensitively(monkeypatch):
    def fake_version(package_name):
        versions = {
            "qwen-asr": "0.0.6",
            "transformers": "5.5.0",
        }
        return versions[package_name]

    def fake_requires(package_name):
        assert package_name == "qwen-asr"
        return ["Transformers==4.57.6"]

    monkeypatch.setattr(qwen3_model, "_package_version", fake_version)
    monkeypatch.setattr(qwen3_model, "_package_requires", fake_requires)

    with pytest.raises(ImportError) as exc_info:
        qwen3_model._check_qwen3_asr_dependencies()

    assert "requires transformers==4.57.6" in str(exc_info.value)


def test_qwen3_asr_dependency_check_does_not_crash_on_non_pep440_transformers(monkeypatch):
    def fake_version(package_name):
        versions = {
            "qwen-asr": "0.0.6",
            "transformers": "custom-main",
        }
        return versions[package_name]

    def fake_requires(package_name):
        assert package_name == "qwen-asr"
        return ["transformers==4.57.6"]

    monkeypatch.setattr(qwen3_model, "_package_version", fake_version)
    monkeypatch.setattr(qwen3_model, "_package_requires", fake_requires)

    qwen3_model._check_qwen3_asr_dependencies()
