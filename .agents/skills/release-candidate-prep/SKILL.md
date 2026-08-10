---
name: release-candidate-prep
description: Preflight and prepare an OpenAI Agents Python release candidate in a dedicated worktree from exact origin/main, gate readiness before branch creation, freeze the released API contract, create one local release commit, enforce final release review as a checker, and produce release-specific PR text. Use only when explicitly invoked with a version. Never push, open a PR, or mutate GitHub.
---

# Release Candidate Preparation

Use this skill only when the user explicitly invokes `$release-candidate-prep` and supplies a release version without a leading `v`, for example `VERSION=0.20.1`. This skill replaces the removed GitHub Actions release-PR creator with a reviewed local workflow.

## Non-negotiable boundaries

- Treat explicit invocation as authorization to fetch `origin/main`, create one dedicated detached release worktree, run branch-free release-readiness gates there, create `release/v<version>` in that worktree only after those gates pass, update the three release-owned files, and create one local commit.
- Keep the user's source checkout on its existing clean `main` commit. Do not fast-forward it, switch its branch, or materialize release files there. Leave the dedicated release worktree in place for green handoff, blocked review, or recoverable failure.
- Never push, open or edit a pull request, add labels or milestones, create a release, or otherwise mutate GitHub. Never run `gh`.
- Own exactly `pyproject.toml`, `uv.lock`, and `tests/fixtures/released_api_contract.json`. Runtime, documentation, workflow, or other repository changes must land on `main` before release preparation.
- Do not stash, reset, delete, overwrite, remove an existing worktree, or work around unrelated local changes. Fail before branch creation when the initial checkout is dirty or is not on `main`, the dedicated worktree is not clean and detached at refreshed `origin/main`, the release branch collides locally or remotely, the prospective packaged-contract gate fails after the allowed dependency-bootstrap recovery, the planning review blocks, or `origin/main` advances after those gates run.
- Treat `$final-release-review` as the controlling release checker, not only as a report generator. Its planning gate must be green before branch creation, and its final-candidate gate must inspect the materialized worktree and be green before PR-ready handoff. Any candidate content, commit, or base change invalidates the previous green result.
- Remove inherited `OPENAI_API_KEY` from every child command. Release preparation does not require a live OpenAI API request.
- Stop after the local commit, final release review, and copy-ready handoff. The user owns the push and pull-request creation.

## 1. Establish the release input

Require one semver-like version without a leading `v`. Do not infer a version from milestones, branch names, or local modifications. Announce that the skill will create and retain a dedicated release worktree with one local commit, keep the source checkout unchanged, and not write to GitHub.

Read `$final-release-review` completely before starting. Its final-candidate report is the release pull request description. Do not use `$pr-draft-summary` for the release candidate itself; this skill owns the fixed release branch, commit subject, title, and description. Continue to use `$pr-draft-summary` normally when implementing changes to this skill or other repository behavior.

## 2. Create an isolated branch-free preflight input

From the repository root, run:

```bash
env -u OPENAI_API_KEY -u GITHUB_TOKEN -u GH_TOKEN UV_DEFAULT_INDEX=https://pypi.org/simple uv run --frozen python .agents/skills/release-candidate-prep/scripts/prepare.py preflight --version <version> --worktree-root <codex-worktree-root>
```

The helper must complete all of these operations or fail with an actionable error while leaving the source checkout on its original `main` commit:

1. Verify the repository root, `main` branch, and clean working tree.
2. Verify that `release/v<version>` does not exist locally or remotely.
3. Fetch `main` into `origin/main` without merging or switching the source checkout.
4. Choose a unique task-oriented path under the configured Codex worktree root. Check both the filesystem and `git worktree list`; never reuse or delete a collision.
5. Create a detached worktree at exact refreshed `origin/main`, then require that worktree to be clean, detached, and at the exact 40-character base commit.
6. Recheck the release-branch collision and require the source checkout to remain clean on `main` at its original commit.
7. Print the exact base commit, unchanged source-checkout commit, planned branch, and dedicated worktree path for both readiness gates and later materialization.

Record the base commit as `<preflight-base>`, the source-checkout commit as `<source-head>`, and the path as `<release-worktree>`. Do not create or switch branches yet. Keep the detached worktree if a later gate blocks so its exact reviewed source remains inspectable.

## 3. Run the branch-free readiness gates

Bootstrap the dedicated worktree before starting either readiness gate:

```bash
env -u OPENAI_API_KEY -u GITHUB_TOKEN -u GH_TOKEN UV_DEFAULT_INDEX=https://pypi.org/simple make sync
```

This dependency installation is mandatory environment preparation, not candidate materialization. It matches the prospective-contract CI job, which installs all optional dependencies before generating the contract. After synchronization, require `<release-worktree>` to remain clean except for ignored environment or `.tmp` output. If synchronization changes a tracked or untracked repository path, stop with that evidence instead of treating the changed checkout as the reviewed source.

Run both gates against exact `<preflight-base>` before materializing any candidate:

1. Start the prospective packaged-contract gate from `<release-worktree>`:

   ```bash
   env -u OPENAI_API_KEY -u GITHUB_TOKEN -u GH_TOKEN UV_DEFAULT_INDEX=https://pypi.org/simple make check-prospective-released-api-contract
   ```

2. Invoke `$final-release-review` from `<release-worktree>` in **pre-release planning** mode with `TARGET=<preflight-base>` and the requested version as the release intent. Require its release-checker result to be **GREEN LIGHT TO SHIP**. Keep the target pinned to the commit rather than allowing a later `origin/main` refresh to change the reviewed source, and require all local source, contract, and package inspection to use the dedicated worktree.

These gates are independent consumers of the same clean source commit. Start the prospective command as a long-running session and perform the read-only planning review while it runs when the execution environment supports overlap. Wait for both results before continuing. If concurrency is unavailable, run them sequentially with the prospective gate first; correctness must not depend on overlap.

If the prospective command reports only that optional dependency modules are unavailable, treat the result as a recoverable environment-bootstrap failure rather than a contract-gate decision. Do not ask the user to choose between synchronization and fixing `main`. Rerun the credential-free `make sync` command, require the worktree to remain clean, and retry the prospective command exactly once. Do not use this recovery for a contract mismatch, packaging or runtime compatibility failure, changed repository path, or any other substantive gate failure.

If dependency synchronization still fails, the prospective command still reports unavailable dependency modules after the single retry, or either gate otherwise fails or blocks, stop without creating `release/v<version>`, leave the source checkout unchanged, retain the detached worktree, and report its path plus the exact failure or the planning review's unblock checklist. Classify a dependency installation failure as environment or dependency setup, a contract-generation mismatch as public-surface or `tests/fixtures/released_api_contract_policy.json` work on `main`, and a packaged compatibility failure by its actual failing source, packaging, platform, or runtime path. A blocked planning review should direct runtime or documentation-timing follow-up to `main` as applicable. Do not continue merely because the review produced a well-formed report.

After both gates pass, require all of the following before materialization:

- The source checkout is still clean on `main` at the same commit it had before preflight.
- `<release-worktree>` is clean except for ignored `.tmp` output, remains detached, and has `HEAD == <preflight-base>`.
- The planning review's green gate applies to `<preflight-base>` and the requested release intent.

## 4. Materialize the uncommitted candidate

Run:

```bash
env -u OPENAI_API_KEY -u GITHUB_TOKEN -u GH_TOKEN UV_DEFAULT_INDEX=https://pypi.org/simple uv run --frozen python .agents/skills/release-candidate-prep/scripts/prepare.py materialize --version <version> --expected-base <preflight-base> --expected-source-head <source-head> --worktree <release-worktree>
```

The helper must complete all of these operations or fail with an actionable error:

1. Repeat the source-root, clean `main`, version, registered-worktree, detached-HEAD, and local/remote release-branch checks.
2. Refresh `origin/main` again without moving the source checkout.
3. Require refreshed `origin/main` and `<release-worktree>` HEAD to equal `<preflight-base>`. If `origin/main` advanced, retain the old detached worktree and rerun preflight plus both readiness gates in a new exact-base worktree.
4. Create `release/v<version>` inside `<release-worktree>` only after the exact-base check passes.
5. Update the single project version declaration in `pyproject.toml`.
6. Run `make sync` with `UV_DEFAULT_INDEX=https://pypi.org/simple`.
7. Run `make update-released-api-contract VERSION=<version>` and then `make check-released-api-contract VERSION=<version>`.
8. Require exactly the three release-owned paths to be modified in `<release-worktree>`, leave them unstaged and uncommitted, and confirm that the source checkout remains unchanged.

If the helper fails after branch creation, preserve its local branch, dedicated worktree, and working-tree evidence. Report the failing command and state rather than guessing whether a partial run is safe to resume. Never remove the worktree as automatic cleanup.

## 5. Review and commit the exact release diff

Run the remaining commands from `<release-worktree>`. Inspect all release-owned files before staging:

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

## 6. Run the final-candidate release review

Invoke `$final-release-review` from `<release-worktree>` in final-candidate mode with the release commit as `TARGET=HEAD`. This invocation is a release checker: it must inspect the complete candidate diff and the actual checked-out `release/v<version>` contents, including `pyproject.toml`, the editable `openai-agents` entry in `uv.lock`, and `tests/fixtures/released_api_contract.json`. The branch, package metadata, lockfile, contract baseline, contract `baseline_commit`, and intended version must agree.

If the review is blocked, stop. Return its unblock checklist, retain the local branch, commit, and worktree for follow-up, and do not present the candidate as PR-ready. A report body does not authorize continuation when the release call is blocked. After any fix, regenerate the API contract when the public surface may have changed, restore a single release commit, and rerun the complete final-candidate review.

The earlier planning review proves that the source commit was ready before branch creation. This final-candidate review remains required because it verifies the materialized branch, version metadata, lockfile, and frozen contract together. Treat its green release call as the handoff gate, then reuse its complete report as the release pull request description; do not substitute the planning report.

## 7. Recheck main freshness

After a green review, fetch `origin main` again without credentials from `<release-worktree>` and compare it with the release commit's parent. If they differ, the candidate is stale. First verify that the branch is clean, has exactly one local commit, and that the commit changes only the three-file release manifest. Rebase that commit onto the new `origin/main` so Git detects any conflicting release metadata. After a clean rebase, move the local release branch back to `origin/main` with a mixed reset, which preserves the rebased release tree as unstaged task-owned changes. Restore only `tests/fixtures/released_api_contract.json` from `origin/main`, rerun `make sync`, run `make check-prospective-released-api-contract`, update and check the released API contract while `HEAD` is the new base, review the exact manifest again, and recreate the single `release: <version>` commit. The base and candidate content changed, so the previous green check is invalid: rerun `$final-release-review` from the worktree and require a new green release call. Repeat until the reviewed commit is exactly one commit ahead of current `origin/main`.

If replay conflicts or another path changes, stop with recoverable evidence. Do not force a resolution that expands the release commit beyond its manifest.

## 8. Produce the release handoff

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

Also report the dedicated worktree path, local branch, commit SHA, parent `origin/main` commit, and the exact three-file manifest outside the copy-ready block. State explicitly that the source checkout was left unchanged, nothing was pushed, and no pull request was created. Leave the worktree in place for the user's handoff.

## Failure behavior

- Preflight or worktree creation failure: leave the source checkout unchanged and do not delete or reuse any colliding worktree.
- Dependency-bootstrap failure: retry unavailable optional dependency setup only as described in the readiness-gate procedure, then retain the detached worktree and return the exact failure if recovery does not succeed.
- Prospective-contract failure after the allowed dependency-bootstrap recovery or blocked planning review: retain the detached worktree, do not create the release branch, and return the exact failure or unblock checklist.
- Materialization failure: retain the worktree and any branch or uncommitted evidence exactly as left by the failing command.
- Blocked final-candidate review: retain the single release commit and worktree, do not call the candidate PR-ready, and return the checker-derived unblock checklist.
- Freshness conflict or unexpected changed path: stop with recoverable worktree evidence rather than forcing a resolution or expanding the release manifest.
