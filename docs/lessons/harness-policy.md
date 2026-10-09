# Lesson: Harness Policy — Never-Rejects, Budgets, and Repair

Lesson D from the cross-app AI convergence (2026-10). The harness
(`src/policy/executor.ts`) is the policy layer every app shares; its
invariants are the reason hosts can put it directly in a callable.

## runTurn never rejects — including when the harness itself throws

`runTurn` returns a typed `HarnessResult`, never throws. Before 0.3.0 that
held for model failures but not for the harness's own injected dependencies:
a rejecting injected `sleep`, `onFallbackExhausted`, or
`onIterationContext` turned the never-rejects contract into an unhandled
rejection inside a Firebase callable. Since 0.3.0 the executor body runs
inside a guarded wrapper: any throw is caught, the partial trace is
preserved, a `harness_threw` entry is appended, and the result resolves as
`{ ok: false, kind: 'harness_threw', ... }` after attempting the host
fallback handler (whose own throw is also caught — the default fallback
question turn is always safe).

Two related hardening points from the same review:

- **No floating promise arms in timeout races.** `sleep(ms).then(() =>
  reject(...))` without a rejection handler is an unhandled rejection when
  the injected sleep rejects; all race arms now route sleep rejections to
  the race's reject.
- **Host seams must map `ok: false` to a failure kind, never to an
  assistant turn.** (Promoted from the closedloop seam comment — see
  [cross-app-contracts.md](cross-app-contracts.md).) The fallback question
  envelope is a harness-internal last resort; a seam that renders it as a
  model turn teaches users to trust a message no model wrote.

## Budgets: per-turn cap, repair cap, and the split

- `maxOutputTokens` bounds every normal request (library default 8_192).
- `repairMaxOutputTokens` (0.3.0) bounds repair requests only, defaulting
  to `maxOutputTokens`. A host doing a 8_192 normal / 65_536 repair split
  sets it once in options; the repair call site is the only place it is
  applied, and it is validated as a positive integer like every other
  numeric option (config failure `repair_output_tokens_invalid`).
- Thinking tokens count against the output budget
  (`usageMetadata.thoughtsTokenCount`), so a "truncated" envelope can be a
  thinking-heavy response: the repair path lowers `thinkingLevel` one step
  (high → medium → low) whenever the failure carried MAX_TOKENS.

## Degeneration and MAX_TOKENS routing

Degeneration handling follows the signatures described in
[schema-and-degeneration.md](schema-and-degeneration.md): conclusive
`ngram_repetition` takes the degeneration path immediately (skips repair;
counts toward the model's degeneration abort, then next iteration, fallback
after the threshold); `max_tokens` alone is only a hint, so MAX_TOKENS
reaches the repair path **only after extraction and validation fail** — a
MAX_TOKENS response carrying a complete valid envelope succeeds normally.

## Transient error classification is deliberately message-parsing

`src/policy/transientErrors.ts` duck-types provider errors (status fields,
string codes, then embedded JSON in the message) rather than importing SDK
error classes. This is intentional, not laziness: the library has zero
runtime dependencies in core, providers embed codes inconsistently across
SDK versions (numeric status, string status, code, and JSON-in-message all
occur in the wild), and the classifier only needs a boolean. It was ported
from courtpuzzle's `demandExtractionProvider` and kept vendor-agnostic.

## What hosts still own

The harness owns retries, fallback, degeneration, and budgets. Hosts own:
gate order (auth, rate limit, entitlement, validation, fencing), prompt
construction, session state, quota semantics (consume-on-attempt vs
on-success — closedloop's per-model-turn caps consume on attempt by
design), and rendering the typed result. See
[firebase-functions-cost.md](firebase-functions-cost.md) for the hosting
side.
