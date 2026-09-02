from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_HUB_URL = "https://www.funasr.com/"


@pytest.mark.parametrize(
    ("readme", "label"),
    (("README.md", "Deployment hub"), ("README_zh.md", "部署中心")),
)
def test_deployment_hub_is_in_readme_primary_navigation(readme: str, label: str) -> None:
    primary_navigation = "\n".join((ROOT / readme).read_text(encoding="utf-8").splitlines()[:40])
    link = f'<a href="{DEPLOYMENT_HUB_URL}">{label}</a>'

    assert primary_navigation.count(link) == 1


@pytest.mark.parametrize(
    ("readme", "community_line"),
    (
        (
            "README.md",
            "Found FunASR useful? [Star the project](https://github.com/modelscope/FunASR) so more builders can find it.",
        ),
        (
            "README_zh.md",
            "FunASR 对你有帮助？欢迎 [Star 项目](https://github.com/modelscope/FunASR)，让更多开发者找到它。",
        ),
    ),
)
def test_quickstart_has_a_concise_community_discovery_entrypoint(
    readme: str, community_line: str
) -> None:
    quickstart = "\n".join((ROOT / readme).read_text(encoding="utf-8").splitlines()[:45])

    assert quickstart.count(community_line) == 1
