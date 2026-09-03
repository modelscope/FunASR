from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def _tracked_entries() -> list[tuple[str, str]]:
    output = subprocess.check_output(
        ["git", "ls-files", "-s", "-z"], cwd=ROOT, text=False
    )
    entries = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0].decode()
        entries.append((mode, path.decode()))
    return entries


def test_tracked_symlinks_are_portable_and_repository_has_no_gitlinks():
    gitlinks = []
    absolute_links = []
    for mode, relative_path in _tracked_entries():
        path = ROOT / relative_path
        if mode == "160000":
            gitlinks.append(relative_path)
        elif mode == "120000" and os.path.isabs(os.readlink(path)):
            absolute_links.append(relative_path)

    assert not gitlinks, f"source distribution must not contain gitlinks: {gitlinks}"
    assert not absolute_links, (
        "source distribution must not contain absolute symlinks: "
        f"{absolute_links}"
    )


def test_git_archive_has_no_absolute_symlinks(tmp_path):
    archive_path = tmp_path / "funasr.tar"
    tree = subprocess.check_output(
        ["git", "write-tree"], cwd=ROOT, text=True
    ).strip()
    with archive_path.open("wb") as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", tree],
            cwd=ROOT,
            check=True,
            stdout=archive,
        )

    with tarfile.open(archive_path) as archive:
        absolute_links = [
            member.name
            for member in archive.getmembers()
            if member.issym() and os.path.isabs(member.linkname)
        ]

    assert not absolute_links, (
        "git archive must be extractable by source installers without external links: "
        f"{absolute_links}"
    )
