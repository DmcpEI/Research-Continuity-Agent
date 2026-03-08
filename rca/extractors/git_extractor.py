"""Read-only extractor for local git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitExtractor:
    """Collect current git state without mutating the repository."""

    def extract(self, path: str | Path) -> dict[str, Any]:
        repo_path = Path(path)
        head = self._run_git(repo_path, ["rev-parse", "HEAD"])
        branch = self._run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
        status = self._run_git(repo_path, ["status", "--short"])
        recent_commits = self._run_git(repo_path, ["log", "--oneline", "-5"])

        content = "\n".join(
            [
                f"Branch: {branch}",
                f"HEAD: {head}",
                "",
                "Status:",
                status or "clean",
                "",
                "Recent commits:",
                recent_commits,
            ]
        ).strip()

        return {
            "title": repo_path.name,
            "content": content,
            "sections": [
                {"heading": "Repository", "text": f"Branch: {branch}\nHEAD: {head}"},
                {"heading": "Status", "text": status or "clean"},
                {"heading": "Recent commits", "text": recent_commits},
            ],
            "metadata": {"kind": "git", "path": str(repo_path), "branch": branch, "head": head},
        }

    @staticmethod
    def _run_git(repo_path: Path, args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown git error"
            raise RuntimeError(f"GitExtractor failed for {repo_path}: {stderr}")
        return result.stdout.strip()
