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
