#!/usr/bin/env python3
"""
Generate a lightweight automated PR review summary (review.md).

This script compares the PR's base and head commits, summarizes changes, runs a few
heuristics (tests updated, docs updated, risky patterns in diffs), and writes a
Markdown report to `review.md` at the repository root.

You can replace this script with Codex CLI invocation in CI (see docs/codex_pr_review.md).
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter, defaultdict
from typing import List, Tuple


def run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out.strip(), err.strip()


def get_base_head() -> Tuple[str, str]:
    base_sha = os.getenv("BASE_SHA", "").strip()
    head_sha = os.getenv("HEAD_SHA", "").strip()
    base_ref = os.getenv("BASE_REF", "").strip()
    head_ref = os.getenv("HEAD_REF", "").strip()

    if base_sha and head_sha:
        return base_sha, head_sha

    # Fallback: try to resolve from refs if shas are not provided
    resolved_base = base_sha
    resolved_head = head_sha

    if not resolved_base and base_ref:
        rc, out, _ = run(["git", "rev-parse", f"origin/{base_ref}"])
        if rc == 0:
            resolved_base = out

    if not resolved_head and head_ref:
        rc, out, _ = run(["git", "rev-parse", "HEAD"])  # current PR SHA
        if rc == 0:
            resolved_head = out

    if resolved_base and resolved_head:
        return resolved_base, resolved_head

    # Last resort: derive merge-base from HEAD and default branch
    rc, default_branch, _ = run(["git", "symbolic-ref", "refs/remotes/origin/HEAD"])
    if rc == 0 and default_branch.startswith("refs/remotes/"):
        default_branch = default_branch.split("/", 2)[-1]
        rc, merge_base, _ = run(["git", "merge-base", f"origin/{default_branch}", "HEAD"])
        if rc == 0:
            return merge_base, run(["git", "rev-parse", "HEAD"])[1]

    raise RuntimeError("Unable to determine base/head for diff")


def git_diff_name_status(base: str, head: str) -> List[Tuple[str, str]]:
    rc, out, err = run(["git", "diff", "--name-status", f"{base}..{head}"])
    if rc != 0:
        raise RuntimeError(f"git diff --name-status failed: {err}")
    changes = []
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            status, path = parts
            changes.append((status.strip(), path.strip()))
    return changes


def git_diff_stat(base: str, head: str) -> str:
    rc, out, _ = run(["git", "diff", "--stat", f"{base}..{head}"])
    return out


def git_diff_unified0(base: str, head: str) -> str:
    rc, out, err = run(["git", "diff", "--unified=0", f"{base}..{head}"])
    if rc != 0:
        return ""
    return out


def top_level(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else path


def main() -> None:
    pr_number = os.getenv("PR_NUMBER", "").strip()
    base, head = get_base_head()

    changes = git_diff_name_status(base, head)
    diff_stat = git_diff_stat(base, head)
    diff_unified0 = git_diff_unified0(base, head)

    changed_files = [p for _, p in changes]
    statuses = Counter(s for s, _ in changes)
    top_levels = Counter(top_level(p) for p in changed_files)

    # Flags and heuristics
    any_py = any(p.endswith(".py") for p in changed_files)
    any_src = any(p.startswith("src/") for p in changed_files)
    any_tests = any(p.startswith("tests/") or p.startswith("scripts/tests/") for p in changed_files)
    any_docs = any(p.startswith("docs/") or p.startswith("documentation/") for p in changed_files)
    main_changed = any(p == "main.py" for p in changed_files)
    workflows_changed = any(p.startswith(".github/") for p in changed_files)

    # Risky patterns from added lines
    risky_patterns = {
        "print(": 0,
        "pdb.set_trace": 0,
        "FIXME": 0,
        "TODO": 0,
        "assert ": 0,
    }
    added_by_file = defaultdict(list)
    current_file = None
    for line in diff_unified0.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        if not line.startswith("+"):
            continue
        if line.startswith("+++ "):
            continue
        if current_file:
            added_by_file[current_file].append(line[1:])
            for pat in risky_patterns:
                if pat in line:
                    risky_patterns[pat] += 1

    # Build review markdown
    lines: List[str] = []
    title = "Automated Review Summary"
    lines.append(f"# {title}")
    if pr_number:
        lines.append(f"PR: #{pr_number}")
    lines.append("")

    lines.append("## Change Overview")
    lines.append(f"- Files changed: {len(changed_files)}")
    lines.append(f"- Change types: {', '.join(f'{k}:{v}' for k, v in sorted(statuses.items())) or 'none'}")
    if top_levels:
        lines.append("- Touched areas: " + ", ".join(f"{k}({v})" for k, v in sorted(top_levels.items())))
    lines.append("")

    if diff_stat:
        lines.append("<details><summary>Diff stat</summary>")
        lines.append("")
        lines.append("```")
        lines.append(diff_stat)
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    # Checks & guidance
    lines.append("## Checks & Guidance")
    if any_src and not any_tests and any_py:
        lines.append("- Tests: src changes detected but no tests updated. Consider adding/updating tests under `tests/`.")
    else:
        lines.append("- Tests: OK (tests updated or non-code changes)")

    if main_changed and not any_docs:
        lines.append("- Docs: CLI `main.py` changed but no docs updated. Consider updating `docs/cli_interface.md`.")
    elif any_src and not any_docs:
        lines.append("- Docs: Source changes without docs updates. If public APIs changed, update docs.")
    else:
        lines.append("- Docs: OK")

    if workflows_changed:
        lines.append("- CI: Workflow files changed. Verify permissions and expected triggers.")

    # Risky patterns summary
    flagged = [f"{k}({v})" for k, v in risky_patterns.items() if v > 0]
    if flagged:
        lines.append("- Risky patterns in added lines: " + ", ".join(flagged))
    else:
        lines.append("- Risky patterns: None detected in added lines")
    lines.append("")

    # Hotspots per file (only if something flagged)
    if flagged:
        lines.append("<details><summary>Flagged lines (by file)</summary>")
        lines.append("")
        for f, added in sorted(added_by_file.items()):
            hits = [ln for ln in added if any(p in ln for p in risky_patterns)]
            if not hits:
                continue
            lines.append(f"- {f}")
            lines.append("  <details><summary>Show lines</summary>")
            lines.append("")
            lines.append("  \n".join(f"  + {h}" for h in hits))
            lines.append("  </details>")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    # Suggested next steps
    lines.append("## Suggested Next Steps")
    lines.append("- Run unit tests: `python -m unittest discover -s tests -v`")
    lines.append("- Run targeted tests if relevant: `python tests/test_vector_database_fix.py` or `python tests/test_extension.py`")
    lines.append("- Retrieval suite (optional): `python scripts/tests/run_retrieval_tests.py --config tests/retrieval_test_prompts.json --output test_results`")
    lines.append("- Verify CLI flows: `python main.py status` and sample `ingest`/`query` commands")
    lines.append("")

    with open("review.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print("Wrote review.md")


if __name__ == "__main__":
    main()

