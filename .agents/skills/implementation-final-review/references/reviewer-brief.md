# Independent Reviewer Brief

Use this template to prepare one self-contained, factual snapshot packet per fingerprint round. Fill every field or mark it explicitly `none` or `not applicable`; do not dispatch an incomplete packet. Fill it once, reuse the shared body byte-for-byte for every reviewer, and vary only the final specialty assignment. Keep this control-plane brief near 12 KB when practical. Store larger evidence in indexed files and reference each file by exact path and SHA-256 digest. Do not omit decision-relevant evidence merely to meet the soft size target. Do not include implementer conclusions, suspected bugs, prior findings, or intended fixes.

## Shared evidence

- Original requirement:
- Implementation scope contract:
  - Required behavior:
  - Compatibility requirements:
  - Intentionally unsupported cases and failure behavior:
  - Supported alternative or `none`:
- Intended target:
- Resolved merge base:
- HEAD:
- Latest release boundary when relevant:
- Risk tier and reason:
- Task-global ledger path, task identity, current round, and remaining authorized budget:
- Canonical task manifest:
- Component manifests:
- Semantic component dependency map and invalidation reasons:
- Combined, component, and repository fingerprints:
- Exact fingerprint revalidation command:
- Raw repository status:
- Complete three-dot diff command:
- Indexed evidence manifest (`ID | exact path | SHA-256 | purpose`):
- Focused preflight commands and results:
- Same-fingerprint verification already credited, or `none`:
- Eligible concurrent final-gate commands and non-mutation basis:
- Gates deferred because they may mutate task-owned content, or `none`:
- Selected architecture references or exact relevant excerpts:

## Contract-surface inventory

Give every row a stable ID. Use one row per changed public symbol, configuration field, event, serialized field, wire value, or documented behavior.

`ID | surface | producers/constructors | consumers/forwarding branches/adapters | default/missing/invalid behavior | package exports/generated public surfaces | adjacent docs/examples | caller-visible tests`

Include adjacent surfaces found outside the current diff. If a required update is absent, add it to the task manifest before freezing the review.

## Await-boundary or authority inventory

For concurrency, cancellation, reentrancy, or lifecycle state:

`ID | operation | state snapshot | await/blocking point | events/operations possible while suspended | monotonic evidence retained | revalidation | side effects/invariant`

Populate supported states including source completion, newer active operation with known or unknown identity, newer operation started then completed, and awaited-action failure or cancellation. If the contract depends on whether something ever happened, identify the monotonic evidence or the serialization proof.

For protocol, security, or persistence instead use:

`ID | input/authority | validation | in-memory state | persisted/serialized state | retry/replay | output | exception/log/telemetry exposure | cleanup/revocation`

## Reviewer instructions

Perform exactly one read-only review round on the frozen fingerprint. Your context must be created with no inherited implementer conversation; the dispatcher uses `fork_turns: "none"` when available. First run the supplied revalidation command and calculate the merge base. Then inspect the complete raw diff, surrounding source, tests, and supplied references. Validate every assigned inventory row rather than trusting the implementer. You may report blockers outside your specialty.

Do not edit or stage files, recursively invoke the review workflow, spawn another reviewer, run broad repository verification, inspect memory, rediscover workflow skills, rerun implementation strategy, search for the fingerprint helper, or rediscover the release tag. Inherit the supplied implementation scope contract; if it is inconsistent or leaves a decision-relevant ambiguity, report that uncertainty to the implementer instead of launching a strategy pass. If any mandatory packet field is neither populated nor explicitly marked `none` or `not applicable`, report the missing field and do not return a creditable clean verdict. Reopen primary source or released evidence only when supplied evidence is inconsistent or leaves a decision-relevant uncertainty; do not use reopening to replace missing packet contents. Run only focused non-mutating probes needed to resolve such uncertainty.

Use approximately 12 source-inspection tool calls as a soft budget. Exceed it whenever decision-relevant uncertainty requires more evidence, but record a concise reason. Do not skip evidence or lower review quality to stay within the budget.

Return exactly one JSON object with this shape and no prose outside it:

```json
{
  "verdict": "clean | findings require fixes | complexity reset required | incomplete packet",
  "reviewed_fingerprints": {
    "combined": "...",
    "components": {"component-name": "..."}
  },
  "checked_inventory_ids": ["..."],
  "unchecked_inventory_ids": [{"id": "...", "reason": "..."}],
  "high_risk_dimensions_checked": ["..."],
  "focused_probes": [{"command": "...", "result": "..."}],
  "remaining_uncertainty": ["..."],
  "findings": [
    {
      "priority": "P0 | P1 | P2 | P3",
      "title": "...",
      "location": "path:line or symbol",
      "failure_scenario": "...",
      "user_consequence": "...",
      "support_basis": "...",
      "baseline_patch_evidence": "... | not applicable",
      "smallest_safe_correction": "...",
      "root_cause_id": "..."
    }
  ],
  "sibling_scenario_scan": [{"root_cause_id": "...", "inventory_ids": ["..."], "result": "..."}],
  "inspection_call_count": 0,
  "inspection_budget_reason": "none | ..."
}
```

Use empty arrays for `focused_probes`, `remaining_uncertainty`, `findings`, or `sibling_scenario_scan` when there are none. Every assigned inventory ID must appear in either `checked_inventory_ids` or `unchecked_inventory_ids`. A `clean` verdict requires an empty `unchecked_inventory_ids`, `remaining_uncertainty`, and `findings` array.

Every `focused_probes[].command` must contain the exact executable command that ran. For a non-shell tool call, provide the complete tool name and arguments. Prose-only labels, omitted arguments, and placeholders such as `<focused probe>` are incomplete and earn no clean credit. If the exact command would be too large to return, place the probe code in an indexed evidence artifact before execution and return its path, SHA-256 digest, and exact execution command.

A bare `clean` or generic checklist is incomplete and earns no clean credit. A malformed JSON object or missing required field is equally incomplete.

## Specialty assignment

- Primary dimensions:
- Required inventory rows:
- Expected component boundaries:
- Evidence items expected to be sufficient:
- Complementary reviewer assignment, if any:
