from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_api"
PLAN = ROOT / "docs" / "community_growth_20k.md"


def test_openai_api_docker_validation_script_documents_cpu_and_gpu_paths():
    script = EXAMPLE / "validate_docker.sh"
    text = script.read_text()

    required = [
        "docker build -t",
        "docker run --rm -d",
        "FUNASR_DEVICE=cpu",
        "python smoke_test.py --base-url",
        "--gpu",
        "--gpus all",
        "FUNASR_DEVICE=cuda",
        "nvidia-smi",
        "NVIDIA Container Toolkit",
    ]
    for marker in required:
        assert marker in text


def test_openai_api_readmes_link_docker_validation_script():
    readmes = [
        EXAMPLE / "README.md",
        EXAMPLE / "README_zh.md",
    ]

    for readme in readmes:
        text = readme.read_text()
        assert "validate_docker.sh" in text
        assert "FUNASR_DEVICE=cpu" in text
        assert "FUNASR_DEVICE=cuda" in text


def test_growth_plan_records_docker_validation_blocker_and_command():
    text = PLAN.read_text()

    required = [
        "[ ] Validate Docker CPU and GPU images",
        "`examples/openai_api/validate_docker.sh`",
        "`bash validate_docker.sh`",
        "`bash validate_docker.sh --gpu`",
        "`ind-gpu8` has H100 GPUs but no `docker` CLI",
        "`funasr-web` also has no `docker` CLI",
    ]
    for marker in required:
        assert marker in text
