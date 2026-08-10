#!/usr/bin/env python3
"""Focused tests for local release candidate preparation."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


class ReleaseRepository:
    """Create a disposable release repository with a local bare origin."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.origin = root / "origin.git"
        self.repo.mkdir()
        run(self.repo, "git", "init", "--initial-branch=main")
        run(self.repo, "git", "config", "user.name", "Release Test")
        run(self.repo, "git", "config", "user.email", "release-test@example.com")
        self._write_fixture_files()
        run(self.repo, "git", "add", ".")
        run(self.repo, "git", "commit", "-m", "Initial release source")
        run(root, "git", "init", "--bare", str(self.origin))
        run(self.repo, "git", "remote", "add", "origin", str(self.origin))
        run(self.repo, "git", "push", "--set-upstream", "origin", "main")
        self.base_commit = run(self.repo, "git", "rev-parse", "HEAD").stdout.strip()

    def advance_origin(self) -> str:
        updater = self.root / "updater"
        run(self.root, "git", "clone", "--branch", "main", str(self.origin), str(updater))
        run(updater, "git", "config", "user.name", "Release Updater")
        run(updater, "git", "config", "user.email", "release-updater@example.com")
        (updater / "new-source.txt").write_text("new source\n", encoding="utf-8")
        run(updater, "git", "add", "new-source.txt")
        run(updater, "git", "commit", "-m", "Advance main")
        run(updater, "git", "push", "origin", "main")
        return run(updater, "git", "rev-parse", "HEAD").stdout.strip()

    def _write_fixture_files(self) -> None:
        (self.repo / "tests/fixtures").mkdir(parents=True)
        (self.repo / "pyproject.toml").write_text(
            '[project]\nname = "openai-agents"\nversion = "0.19.4"\n',
            encoding="utf-8",
        )
        (self.repo / "uv.lock").write_text(
            'version = 1\n\n[[package]]\nname = "openai-agents"\nversion = "0.19.4"\n'
            'source = { editable = "." }\n',
            encoding="utf-8",
        )
        (self.repo / "tests/fixtures/released_api_contract.json").write_text(
            json.dumps(
                {
                    "baseline": "v0.19.4",
                    "baseline_commit": "a" * 40,
                    "callables": {},
                    "required_top_level_exports": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.repo / "fake_release.py").write_text(
            """from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
action = sys.argv[1]
if {'GH_TOKEN', 'GITHUB_TOKEN', 'OPENAI_API_KEY'} & os.environ.keys():
    raise SystemExit('credentials must not reach release subprocesses')
if os.environ.get('UV_DEFAULT_INDEX') != 'https://pypi.org/simple':
    raise SystemExit('UV_DEFAULT_INDEX must use the public package index')
version = re.search(
    r'^version = \"([^\"]+)\"$',
    (root / 'pyproject.toml').read_text(),
    re.MULTILINE,
).group(1)
if action == 'sync':
    path = root / 'uv.lock'
    text = path.read_text()
    text = re.sub(
        r'(name = \"openai-agents\"\\nversion = \")[^\"]+(\")',
        rf'\\g<1>{version}\\g<2>',
        text,
    )
    path.write_text(text)
elif action == 'update':
    expected = sys.argv[2]
    if expected != version:
        raise SystemExit('version mismatch')
    path = root / 'tests/fixtures/released_api_contract.json'
    contract = json.loads(path.read_text())
    contract['baseline'] = f'v{version}'
    contract['baseline_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True
    ).strip()
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\\n')
elif action == 'check':
    expected = sys.argv[2]
    contract = json.loads(
        (root / 'tests/fixtures/released_api_contract.json').read_text()
    )
    if expected != version or contract['baseline'] != f'v{version}':
        raise SystemExit('contract mismatch')
else:
    raise SystemExit(f'unknown action: {action}')
""",
            encoding="utf-8",
        )
        (self.repo / "Makefile").write_text(
            "sync:\n\tpython fake_release.py sync\n\n"
            "update-released-api-contract:\n\tpython fake_release.py update $(VERSION)\n\n"
            "check-released-api-contract:\n\tpython fake_release.py check $(VERSION)\n",
            encoding="utf-8",
        )


class VersionTests(unittest.TestCase):
    def test_validate_version_accepts_release_and_prerelease(self) -> None:
        self.assertEqual(prepare.validate_version("0.20.1"), "0.20.1")
        self.assertEqual(prepare.validate_version("0.21.0-rc1"), "0.21.0-rc1")

    def test_validate_version_rejects_ambiguous_values(self) -> None:
        for value in ("v0.20.1", "0..20.1", "next", "0.20.1/other"):
            with self.subTest(value=value), self.assertRaises(prepare.ReleasePreparationError):
                prepare.validate_version(value)

    def test_replace_project_version_requires_exactly_one_change(self) -> None:
        text = '[project]\nversion = "0.19.4"\n'
        self.assertEqual(
            prepare.replace_project_version_text(text, "0.20.0"),
            '[project]\nversion = "0.20.0"\n',
        )
        with self.assertRaises(prepare.ReleasePreparationError):
            prepare.replace_project_version_text(text, "0.19.4")
        with self.assertRaises(prepare.ReleasePreparationError):
            prepare.replace_project_version_text(
                text + '[tool.example]\nversion = "1.0.0"\n',
                "0.20.0",
            )


class PreparationTests(unittest.TestCase):
    def test_prepare_creates_branch_and_exact_uncommitted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseRepository(Path(directory))

            with mock.patch.dict(
                os.environ,
                {
                    "GH_TOKEN": "untrusted",
                    "GITHUB_TOKEN": "untrusted",
                    "OPENAI_API_KEY": "untrusted",
                },
            ):
                candidate = prepare.prepare(fixture.repo, "0.20.0")

            self.assertEqual(candidate.base_commit, fixture.base_commit)
            self.assertEqual(candidate.branch, "release/v0.20.0")
            self.assertEqual(set(candidate.changed_paths), prepare.RELEASE_PATHS)
            self.assertEqual(
                run(fixture.repo, "git", "branch", "--show-current").stdout.strip(),
                "release/v0.20.0",
            )
            self.assertEqual(
                run(fixture.repo, "git", "rev-list", "--count", "origin/main..HEAD").stdout.strip(),
                "0",
            )
            remote_heads = run(
                fixture.repo,
                "git",
                "ls-remote",
                "--heads",
                "origin",
                "release/v0.20.0",
            ).stdout
            self.assertEqual(remote_heads, "")
            contract = json.loads(
                (fixture.repo / "tests/fixtures/released_api_contract.json").read_text()
            )
            self.assertEqual(contract["baseline"], "v0.20.0")
            self.assertEqual(contract["baseline_commit"], fixture.base_commit)

    def test_prepare_rejects_dirty_main_without_creating_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseRepository(Path(directory))
            (fixture.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

            with self.assertRaisesRegex(
                prepare.ReleasePreparationError,
                "clean working tree",
            ):
                prepare.prepare(fixture.repo, "0.20.0")

            self.assertEqual(
                run(fixture.repo, "git", "branch", "--show-current").stdout.strip(),
                "main",
            )

    def test_prepare_fast_forwards_to_refreshed_origin_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseRepository(Path(directory))
            refreshed_base = fixture.advance_origin()

            candidate = prepare.prepare(fixture.repo, "0.20.0")

            self.assertEqual(candidate.base_commit, refreshed_base)
            self.assertEqual(
                run(fixture.repo, "git", "rev-parse", "HEAD").stdout.strip(),
                refreshed_base,
            )
            self.assertTrue((fixture.repo / "new-source.txt").is_file())

    def test_prepare_rejects_existing_remote_release_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseRepository(Path(directory))
            run(
                fixture.repo,
                "git",
                "push",
                "origin",
                "HEAD:refs/heads/release/v0.20.0",
            )

            with self.assertRaisesRegex(
                prepare.ReleasePreparationError,
                "Remote branch 'release/v0.20.0' already exists",
            ):
                prepare.prepare(fixture.repo, "0.20.0")

            self.assertEqual(
                run(fixture.repo, "git", "branch", "--show-current").stdout.strip(),
                "main",
            )


if __name__ == "__main__":
    unittest.main()
