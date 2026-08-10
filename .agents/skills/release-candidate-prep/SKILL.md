---
name: release-candidate-prep
description: Prepare an OpenAI Agents Python release candidate locally from exact origin/main, freeze the released API contract, create one local release commit, run final release review, and produce release-specific PR text. Use only when explicitly invoked with a version. Never push, open a PR, or mutate GitHub.
---

# Release Candidate Preparation

Use this skill only when the user explicitly invokes `$release-candidate-prep` and supplies a release version without a leading `v`, for example `VERSION=0.20.1`. This skill replaces the removed GitHub Actions release-PR creator with a reviewed local workflow.

## Non-negotiable boundaries

- Treat explicit invocation as authorization to fast-forward a clean local `main`, create `release/v<version>`, update the three release-owned files, and create one local commit.
- Never push, open or edit a pull request, add labels or milestones, create a release, or otherwise mutate GitHub. Never run `gh`.
- Own exactly `pyproject.toml`, `uv.lock`, and `tests/fixtures/released_api_contract.json`. Runtime, documentation, workflow, or other repository changes must land on `main` before release preparation.
- Do not stash, reset, delete, overwrite, or work around unrelated local changes. Fail before branch creation when the initial checkout is dirty, is not on `main`, has diverged from refreshed `origin/main`, or collides with a local or remote release branch.
- Remove inherited `OPENAI_API_KEY` from every child command. Release preparation does not require a live OpenAI API request.
- Stop after the local commit, final release review, and copy-ready handoff. The user owns the push and pull-request creation.

## 1. Establish the release input

Require one semver-like version without a leading `v`. Do not infer a version from milestones, branch names, or local modifications. Announce that the skill will update the local checkout and create one commit but will not write to GitHub.

Read `$final-release-review` completely before starting. Its final-candidate report is the release pull request description. Do not use `$pr-draft-summary` for the release candidate itself; this skill owns the fixed release branch, commit subject, title, and description. Continue to use `$pr-draft-summary` normally when implementing changes to this skill or other repository behavior.

## 2. Prepare the uncommitted candidate

From the repository root, run:

```bash
env -u OPENAI_API_KEY -u GITHUB_TOKEN -u GH_TOKEN UV_DEFAULT_INDEX=https://pypi.org/simple uv run --frozen python .agents/skills/release-candidate-prep/scripts/prepare.py --version <version>
```

The helper must complete all of these operations or fail with an actionable error:

1. Verify the repository root, `main` branch, and clean working tree.
2. Verify that `release/v<version>` does not exist locally or remotely.
3. Fetch `main` into `origin/main`, fast-forward local `main` with `git merge --ff-only origin/main`, and require local `HEAD` to equal refreshed `origin/main`.
4. Create `release/v<version>`.
5. Update the single project version declaration in `pyproject.toml`.
6. Run `make sync` with `UV_DEFAULT_INDEX=https://pypi.org/simple`.
7. Run `make update-released-api-contract VERSION=<version>` and then `make check-released-api-contract VERSION=<version>`.
8. Require exactly the three release-owned paths to be modified and leave them unstaged and uncommitted.

If the helper fails after branch creation, preserve its local branch and working-tree evidence. Report the failing command and state rather than guessing whether a partial run is safe to resume.

## 3. Review and commit the exact release diff

Inspect all release-owned files before staging:

```bash
git status --short
git diff --check
git diff -- pyproject.toml uv.lock tests/fixtures/released_api_contract.json
```

Confirm all of the following:

- `pyproject.toml` and the editable `openai-agents` entry in `uv.lock` declare the requested version.
- The API contract baseline is `v<version>` and its `baseline_commit` is the exact `origin/main` source commit on which the release branch is based.
- The generated contract preserves the previous release and freezes intended new exports and signatures.
- Any intended `public_properties`, `canonical_imports`, or `public_modules` policy additions have been reviewed explicitly; the updater deliberately does not infer them.
- No path outside the three-file release manifest is changed, staged, or untracked.

Stage only the manifest and create exactly one local commit:

```bash
git add pyproject.toml uv.lock tests/fixtures/released_api_contract.json
git commit -m "release: <version>"
```

Do not amend unrelated content into the commit.

## 4. Run the final-candidate release review

Invoke `$final-release-review` in final-candidate mode with the release commit as `TARGET=HEAD`. The branch, `pyproject.toml`, `uv.lock`, and API contract must agree on the intended version.

If the review is blocked, stop. Return its unblock checklist, retain the local branch and commit for follow-up, and do not present the candidate as PR-ready. After any fix, regenerate the API contract when the public surface may have changed, restore a single release commit, and rerun the complete final-candidate review.

## 5. Recheck main freshness

After a green review, fetch `origin main` again without credentials and compare it with the release commit's parent. If they differ, the candidate is stale. First verify that the branch is clean, has exactly one local commit, and that the commit changes only the three-file release manifest. Rebase that commit onto the new `origin/main` so Git detects any conflicting release metadata. After a clean rebase, move the local release branch back to `origin/main` with a mixed reset, which preserves the rebased release tree as unstaged task-owned changes. Restore only `tests/fixtures/released_api_contract.json` from `origin/main`, rerun `make sync`, update and check the API contract while `HEAD` is the new base, review the exact manifest again, and recreate the single `release: <version>` commit. Then rerun `$final-release-review`. Repeat until the reviewed commit is exactly one commit ahead of current `origin/main`.

If replay conflicts or another path changes, stop with recoverable evidence. Do not force a resolution that expands the release commit beyond its manifest.

## 6. Produce the release handoff

For a green, current candidate, return the `$final-release-review` report plus this release-specific block in English:

```markdown
# Release Pull Request

## Branch

release/v<version>

## Commit

release: <version>

## Title

Release <version>

## Description

<the complete final-candidate report from $final-release-review>
```

Apply the repository's GitHub paste-readiness rules to the report. Use native `#123` references for this repository and `owner/repo#123` for another repository. Keep the required compare URL. Do not include local paths, Codex citations, operational diagnostics, or app directives inside the copy-ready description.

Also report the local branch, commit SHA, parent `origin/main` commit, and the exact three-file manifest outside the copy-ready block. State explicitly that nothing was pushed and no pull request was created.
