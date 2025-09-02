# Automated PR Review (Codex-style)

This repository includes a GitHub Actions workflow that posts an automated review comment on every pull request. By default it runs a lightweight Python script that summarizes the diff and flags common risks. You can swap this step for Codex CLI to generate a richer review.

## What You Get
- Single PR comment titled “Automated Review Summary”
- Changed files overview and diff stat
- Checks for tests/docs coverage when source changes
- Simple risk scan in added lines (e.g., `print(`, `pdb.set_trace`, `TODO`, `FIXME`)
- Quick commands to run tests locally

## Added Files
- `.github/workflows/codex-pr-review.yml` – CI workflow triggered on PR events
- `scripts/ci/generate_review.py` – Lightweight review generator that writes `review.md`

## How It Works
1. On PR events (opened/synchronize/reopened), the workflow checks out the repo and runs `scripts/ci/generate_review.py`.
2. The script compares the PR branch to the base commit and writes `review.md` with a summary and guidance.
3. The workflow posts/updates a PR comment with the contents of `review.md`.

Permissions: the workflow uses the default `GITHUB_TOKEN` with `pull-requests: write` and `contents: read`.

## Swap In Codex CLI (Optional)
If you have Codex CLI available in your workflow runner, replace the “Generate review” step with your Codex invocation that writes `review.md`.

Example pattern:
- Set a repository variable `CODEX_CMD` to your Codex command (must write `review.md` at repo root), e.g.:
  - `codex pr-review --base "$BASE_SHA" --head "$HEAD_SHA" --out review.md`
- Uncomment the optional step in `.github/workflows/codex-pr-review.yml` and ensure it runs instead of the lightweight script.

Notes:
- The workflow already exports `BASE_SHA`, `HEAD_SHA`, `BASE_REF`, `HEAD_REF`, and `PR_NUMBER` to the review step.
- Ensure your Codex command exits non‑zero on failure so the job fails visibly.

## Local Testing
You can run the script locally before opening a PR:
- Ensure your working tree is clean and you have the base branch fetched.
- Run: `BASE_REF=main HEAD_REF=$(git rev-parse HEAD) python scripts/ci/generate_review.py`
- Open `review.md` to preview the comment content.

## Troubleshooting
- No comment appears: verify the workflow ran and has `pull-requests: write` permission.
- Empty diff: ensure `actions/checkout` is using `fetch-depth: 0` (already configured).
- Too many comments: the workflow updates an existing comment if found; otherwise it creates one. Each push should replace the prior comment.

## Customization Ideas
- Expand heuristics in `scripts/ci/generate_review.py` to enforce repo‑specific checks.
- Attach artifacts (HTML report) in addition to the comment.
- Gate merges on review quality by turning warnings into job failures.

