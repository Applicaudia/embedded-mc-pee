# Lesson: Schema-in-Prompt and Degeneration Detection

Lessons A and B from the cross-app AI convergence (2026-10). Sources: ADR-0001,
the Gemini 3.x probe matrix in [../gemini-debugging.md](../gemini-debugging.md),
and production behavior in closedloop and courtpuzzle.

## Lesson A: Never send `responseSchema` (schema-in-prompt instead)

Gemini 3.x Flash models token-repetition-loop to `MAX_TOKENS` when a large
`responseSchema` constrains the response. This was observed repeatedly in
production (courtpuzzle extraction providers, closedloop early experiments) and
confirmed by probe: `responseSchema` + `thinkingLevel` together degenerate,
while the same payload shape requested via prompt text with runtime validation
comes back clean.

The belt-and-suspenders pattern (ADR-0001) is the library's core contract
mechanism:

1. One manifest per payload surface (`createContract(manifest, validator)`).
2. The manifest renders the schema as **prompt text** in the system
   instruction — `responseSchema` is never set anywhere in the library
   (`src/gemini/geminiTransport.ts` carries the CRITICAL comment).
3. Runtime validation enforces the shape on extraction; validation failure
   routes to repair, then fallback — never to an untyped success.

Rule of thumb: the model is told what shape to produce; the code proves it.
Prompt contracts degrade gracefully (the model usually complies), while
`responseSchema` constraints degrade catastrophically (repetition loops that
burn the entire token budget).

courtpuzzle keeps an eslint `no-restricted-syntax` ban on `responseSchema` as
a permanent regression guard; its extraction providers are scheduled to
migrate onto this library's manifest mechanism (epic plan, Phase 6).

## Lesson B: Detect degeneration by signature, not by finishReason alone

Degenerate output has two distinct signatures in `src/diagnostics/degeneration.ts`:

- `ngram_repetition` — **conclusive**. A phrase of 1-8 words repeats
  consecutively until the total reaches the threshold (default 120 words).
  The period is detected, not assumed: observed loops repeat short phrases
  ("Done. Complete. OK.") and single words alike. A conclusive signature
  takes the degeneration path immediately (skips repair; counts toward the
  model's degeneration abort, decision `repair_skipped_degenerate`), and
  the model falls back after the abort threshold.
- `max_tokens` — **suggestive only** since 0.3.0. A `finishReason:
  'MAX_TOKENS'` response can still carry a complete, valid envelope, so
  MAX_TOKENS reaches the degeneration/repair path **only after extraction
  and validation fail**. Treating MAX_TOKENS as automatically degenerate
  skips repair on responses that were fine.

Two related traps:

- **Thinking tokens count against the output budget.**
  `usageMetadata.thoughtsTokenCount` is billed inside `maxOutputTokens`, so
  a "truncated" envelope may simply be a thinking-heavy response that
  exhausted the combined budget. Raise the cap or lower `thinkingLevel` on
  retry (the executor's repair path does exactly this).
- **Repair needs headroom.** Repairs re-emit the full envelope, so the
  repair request uses `repairMaxOutputTokens` (default: same as
  `maxOutputTokens`) and lowers `thinkingLevel` one step when the failure
  carried MAX_TOKENS — otherwise the repair hits the same ceiling.

See [model-config.md](model-config.md) for the model-chain and sampling
parameter lessons, and [../gemini-debugging.md](../gemini-debugging.md) for
the probe matrix.
