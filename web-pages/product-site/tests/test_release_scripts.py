from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY = REPO_ROOT / 'web-pages' / 'scripts' / 'deploy-product-site.sh'
ROLLBACK = REPO_ROOT / 'web-pages' / 'scripts' / 'rollback-product-site.sh'
CONVERSION_MAP = REPO_ROOT / 'web-pages' / 'nginx' / 'conversion-map.conf'
NGINX_CONFIG = REPO_ROOT / 'web-pages' / 'nginx' / 'funasr.com.conf'
EXPECTED_FIXED_ROUTES = {
    '/go/github',
    '/go/fun-asr',
    '/go/sensevoice',
    '/go/funclip',
    '/go/docs',
    '/go/releases',
    '/go/deploy-vllm',
    '/go/deploy-llama-cpp',
}
EXPECTED_REPOSITORY_ROUTES = {
    '/go/fun-asr': 'https://github.com/QwenAudio/Fun-ASR',
    '/go/sensevoice': 'https://github.com/QwenAudio/SenseVoice',
    '/go/funclip': 'https://github.com/modelscope/FunClip',
}


def executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding='utf-8')
    path.chmod(0o755)
    return path


def release_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    server = tmp_path / 'server'
    server.mkdir()
    nginx_config = tmp_path / 'nginx.conf'
    nginx_config.write_text('events {}\nhttp {}\n', encoding='utf-8')
    fake_validator = executable(
        tmp_path / 'validate.py',
        'from pathlib import Path\nimport sys\nraise SystemExit(0 if (Path(sys.argv[1]) / "deployment-manifest.json").is_file() else 1)\n',
    )
    fake_nginx = executable(tmp_path / 'nginx', '#!/bin/sh\nexit 0\n')
    fake_curl = executable(tmp_path / 'curl', '#!/bin/sh\nexit 0\n')
    environment = os.environ.copy()
    environment.update({
        'SITE_BASE': str(server),
        'VALIDATOR': str(fake_validator),
        'NGINX_CONFIG': str(nginx_config),
        'NGINX_BIN': str(fake_nginx),
        'CURL_BIN': str(fake_curl),
        'SMOKE_BASE_URL': 'https://www.funasr.com',
    })
    return environment, server


def output_dir(tmp_path: Path, name: str) -> Path:
    output = tmp_path / name
    output.mkdir()
    (output / 'index.html').write_text(name, encoding='utf-8')
    (output / 'deployment-manifest.json').write_text('{"pages": [], "assets": {}}\n', encoding='utf-8')
    return output


def run(script: Path, *arguments: str, environment: dict[str, str]) -> None:
    subprocess.run(
        [str(script), *arguments],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )


def test_deploy_keeps_releases_and_switches_symlink(tmp_path):
    environment, server = release_environment(tmp_path)
    first = output_dir(tmp_path, 'first')
    second = output_dir(tmp_path, 'second')

    run(DEPLOY, str(first), '20260726T120000Z', environment=environment)
    run(DEPLOY, str(second), '20260726T130000Z', environment=environment)

    assert sorted(path.name for path in (server / 'releases').iterdir()) == [
        '20260726T120000Z',
        '20260726T130000Z',
    ]
    assert (server / 'current').resolve().name == '20260726T130000Z'
    assert (server / 'releases/20260726T120000Z/index.html').read_text() == 'first'
    assert len(list((server / 'nginx-backups').glob('nginx.conf.*'))) == 2


def test_rollback_switches_to_retained_release(tmp_path):
    environment, server = release_environment(tmp_path)
    first = output_dir(tmp_path, 'first')
    second = output_dir(tmp_path, 'second')
    run(DEPLOY, str(first), '20260726T120000Z', environment=environment)
    run(DEPLOY, str(second), '20260726T130000Z', environment=environment)

    run(ROLLBACK, '20260726T120000Z', environment=environment)

    assert (server / 'current').resolve().name == '20260726T120000Z'
    assert sorted(path.name for path in (server / 'releases').iterdir()) == [
        '20260726T120000Z',
        '20260726T130000Z',
    ]


def test_invalid_release_id_is_rejected_before_copy(tmp_path):
    environment, server = release_environment(tmp_path)
    output = output_dir(tmp_path, 'output')
    result = subprocess.run(
        [str(DEPLOY), str(output), '../bad'],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert not (server / 'releases').exists()


def test_conversion_map_contains_only_fixed_redirects():
    routes = {}
    for route, target in re.findall(r'^\s*(/go/[\w-]+)\s+(https://[^;]+);', CONVERSION_MAP.read_text(), re.M):
        routes[route] = target

    assert set(routes) == EXPECTED_FIXED_ROUTES
    assert EXPECTED_REPOSITORY_ROUTES.items() <= routes.items()
    assert all(
        target.startswith('https://github.com/') or target.startswith('https://www.funasr.com/')
        for target in routes.values()
    )
    assert '$arg_' not in CONVERSION_MAP.read_text(encoding='utf-8')


def test_nginx_contract_uses_current_release_and_hardening():
    config = NGINX_CONFIG.read_text(encoding='utf-8')

    assert 'root /root/FunASR/web-pages/current;' in config
    assert 'ssl_protocols TLSv1.2 TLSv1.3;' in config
    assert 'Content-Security-Policy' in config
    assert 'Permissions-Policy' in config
    assert 'funasr-conversions.log' in config
    assert r'\.[0-9a-f]{12}' in config
    assert 'max-age=31536000, immutable' in config
