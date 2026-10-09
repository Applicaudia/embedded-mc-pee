# Lesson: Model Chains and Sampling Parameters

Lessons C and H from the cross-app AI convergence (2026-10). Trigger: Google
deprecation notice (2026-10-09) — upcoming Gemini models hard-400
`thinking_budget` and **error on `temperature` / `top_p` / `top_k`**.

## Lesson C: The model chain is a library SSOT, not a host constant

Every app independently hand-rolled its own fallback model list, and they
drifted: retired models pinned as primary in one app, aliases in positions
where a pinned version belongs in another. The fix landed in 0.3.0:

```ts
import { RECOMMENDED_FLASH_MODEL_CHAIN } from 'embedded-mc-pee';
// ['gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-flash-latest']
```

`src/policy/modelChain.ts` is the single definition (readonly tuple +
rationale comment). Rationale, in order:

- `gemini-3.8-flash` primary — GA, measurably reduced loop-spiraling.
- `gemini-3.7-flash` second — proven fallback.
- 3.5/3.6 dropped — superseded, no reason to burn a fallback slot.
- `gemini-flash-latest` terminal only — an alias must never be primary
  (an alias silently retargets; a pinned version fails loudly when retired).

Host seams adopt it as `models: [...(opts.models ?? RECOMMENDED_FLASH_MODEL_CHAIN)]`
— import, never re-declare — and each seam's unit test asserts its default
equals the library export.

## Lesson H: Stop sending sampling parameters entirely

`temperature` is **never transmitted** since 0.3.0: the executor no longer
applies or forwards it, the Gemini transport no longer maps it, and it is
excluded from the record/replay hash (`DEFAULT_REPLAY_RECORDING_VERSION`
bumped 1 → 2 so stale recordings are rejected instead of silently
live-calling in passthrough lanes). The request fields stay `@deprecated`
with never-transmitted JSDoc notes for one release cycle. `top_p`/`top_k`
were confirmed absent from all call sites.

Why drop rather than default: the deprecation makes the parameter an error on
upcoming models, observed behavior across apps never depended on it (every
call site used a fixed 0.x value copied from a tutorial), and schema-in-prompt
determinism comes from the contract, not from sampling nudges.

### Call-site inventory at removal time (verified 2026-10-09)

Non-test files setting temperature values, per app:

| App | Files | Location |
|---|---|---|
| closedloop | 15 | 11 in `functions/src/callables/` (grill, extraction, review surfaces) + `extractReviewResult.ts`, `extractTemplateQuestions.ts`, `pdfVisionFallback.ts` + the seam `functions/src/shared/ai/emcpHarness.ts` (forwards `opts.temperature` into harness options) |
| strobopro | 1 | `functions/src/shared/geminiClient.ts` |
| courtpuzzle | 3 | `allocationExtractionProvider.ts`, `copilotProvider.ts`, `demandExtractionProvider.ts` |

The seam row matters most: `emcpHarness.ts` is the one file every callable
goes through, so its temperature forwarding must go in the same Phase 2
pass that strips the per-callable values.

All are removed in each app's migration phase (closedloop Phase 2, strobopro
Phase 3, courtpuzzle Phase 6 of the convergence epic).

### thinkingLevel: narrow union, lowercase wire format

The request union is `'low' | 'medium' | 'high'` (`HarnessThinkingLevel`).
`'minimal'` 400s on 3.7/3.8 and was removed everywhere including the
executor's config validator (a dead-accepted value in a validator is worse
than none: it passes config and fails at the provider).

Per-model validity varies (some models reject `medium`; check the model's
docs before raising the default). The REST wire format is the lowercase
names — the `@google/genai` SDK's exported `ThinkingLevel` enum uses
UPPERCASE member names but passes `thinkingConfig` through verbatim, so the
transport sends the lowercase string documented at
ai.google.dev/gemini-api/docs/generate-content (production-verified).

Related: [schema-and-degeneration.md](schema-and-degeneration.md) covers the
repair-time thinkingLevel step-down; [harness-policy.md](harness-policy.md)
covers the repair token budget.
