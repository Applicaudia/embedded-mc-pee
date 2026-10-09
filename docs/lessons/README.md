# Lessons Hub

Cross-app AI lessons learned while building and running this library and
its three host applications (closedloop, strobopro, courtpuzzle). This is
the promotion home for shared AI lessons across Applicaudia apps — a
lesson that applies to more than one AI surface lands here, not in an
app's private docs.

## Index

| File | Lessons | What it covers |
|---|---|---|
| [schema-and-degeneration.md](schema-and-degeneration.md) | A + B | Never send `responseSchema` (ADR-0001 schema-in-prompt); degeneration signatures (`ngram_repetition` conclusive, `max_tokens` suggestive); thinking tokens count against the output budget |
| [model-config.md](model-config.md) | C + H | `RECOMMENDED_FLASH_MODEL_CHAIN` SSOT; temperature/top_p/top_k never sent (with the verified call-site inventory at removal time); `thinkingLevel` narrow union + lowercase wire format |
| [harness-policy.md](harness-policy.md) | D | runTurn never-rejects (incl. `harness_threw`); no floating race arms; `repairMaxOutputTokens` budget split; transient error message-parsing rationale |
| [firebase-functions-cost.md](firebase-functions-cost.md) | E | v2 single-bundle colocation; maxInstances ceiling + wiring tests; `setGlobalOptions` import-order contract; gate order; per-surface budgets; zero-data skip; stubbed/tag-gated E2E |
| [grounding-and-fencing.md](grounding-and-fencing.md) | F | `escapeFenceDelimiters` + DATA ONLY fences for every untrusted block; grounding stays host-side; nested-fence rejection in extraction |
| [cross-app-contracts.md](cross-app-contracts.md) | G | one manifest per surface, states chosen per surface; seam failure contract (`ok: false` → `failed`, never an assistant turn); reject-structural/clamp-cosmetic validators; model chain history; migration honesty |

## Full docs index

- [Architecture](../architecture.md) — module map, data flow, determinism principles
- [TypeScript Guidelines](../guidelines-typescript.md) — coding standards, type-safety discipline
- [ADR-0001: Belt-and-Suspenders Contracts](../adr/0001-belt-and-suspenders-contracts.md) — why schema-in-prompt over responseSchema
- [Gemini Debugging](../gemini-debugging.md) — troubleshooting degeneration loops (probe matrix)
- [Resilience & Clarification](../resilience-and-clarification.md) — proposal-only mutation boundary
- [MCP Contract Mapping](../mcp-contract-mapping.md) — transport duties
- [Technical Debt Database](../tech-debt-db.md) — recorded debt and plans

## Adding a lesson

A lesson belongs here when it is true for more than one AI surface or
more than one host app. Single-app lessons stay in that app's own docs;
library debt goes to [../tech-debt-db.md](../tech-debt-db.md). Every
claim must match actual code behavior — cite the module that implements
it, and update the lesson when the code changes.
