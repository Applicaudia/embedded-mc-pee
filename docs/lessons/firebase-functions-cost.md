# Lesson: Hosting LLM Harnesses on Firebase Functions Without Cost Surprises

Lesson E from the cross-app AI convergence (2026-10), grounded in
closedloop's production Functions setup (`closedloop_web/functions`). The
library is transport-agnostic; these are the hosting-side invariants that
keep an AI-bearing Functions deployment cheap and bounded.

## v2 single-bundle colocation

All callables build into one bundle (`functions/package.json` main:
`dist/combined-index.js`) and deploy as Gen 2 (`firebase-functions/v2`)
functions from a single codebase. Colocation matters for an AI harness: the
module graph (this library, prompts, contracts, Admin SDK) is paid once per
cold start per instance, not once per function. The flip side: global
options must be set before any handler module constructs endpoints — see
import-order contract below.

## Instance ceiling, memory floor, concurrency

closedloop's policy SSOT (`src/sharedSrc/src/constants/functionsPolicy.ts`,
TD-070) is the reference shape:

- **Global `maxInstances` ceiling (10)** caps burst spend; per-request
  abuse controls (per-uid daily turn caps) bound usage long before the
  ceiling matters. Scheduled sweeps are singletons (1 — lease-guarded
  already serializes them); webhooks keep an explicit 10.
- **Memory baseline 512MiB, not the 256MiB default**: the bundled module
  graph's cold-start init alone exceeded 256MiB (createDevPlanRequests OOM
  on QA at 263MiB at container start). Per-function overrides may only
  exceed the baseline.
- **Concurrency stays at the platform default (80 when CPU >= 1)** for all
  handlers: they are I/O-bound (Gemini, Firestore) and memory-per-request
  is small relative to the tier. A handler needing a tighter ceiling
  imports a constant from the policy module, documents why, and extends
  the wiring test.
- **Wiring test**: `functionPolicy.wiring.test.ts` asserts the deployed
  options match the constants, so a drift between policy and wiring fails
  CI rather than prod.

## setGlobalOptions is an import-order contract

`functions/src/shared/functionOptions.ts` calls `setGlobalOptions({region,
maxInstances, memory})` and **must be imported first** in the functions
index: handlers read global options when their module constructs endpoints
(an import-time side effect), so a `setGlobalOptions` placed after handler
imports silently applies to nothing. Single region for every function
(`FUNCTIONS_REGION`), keeping Firestore + function co-location cheap.

## Gate order before the model turn

closedloop's callable chain, verified in `grill1on1.ts`:
**auth → rate limit → entitlement → validate → fence → model**. The rate
limit deliberately precedes entitlement: an unauthorized caller burning a
quota check is cheap; an entitlement check amplifying into self-heal
writes is not. Validation (typed input parsers) precedes any model spend,
and fencing is the last step before prompt assembly.

## Per-surface budgets and zero-data skip

- Every AI surface sets its own turn budget (closedloop `AI_TURN_OPTIONS`:
  `maxOutputTokens` per turn), plus per-uid daily caps enforced in
  Firestore transactions before the model is ever called.
- **Skip the model when there is nothing to extract**: surfaces that
  conditionally need AI check first and return early with a logged reason
  (closedloop: `proposeReviewBriefThemes` skips the theme turn when the
  reviewee has no submitted responses; `repairManualReviewParse` skips
  when deterministic parse already resolved every question). A model call
  with zero input data is pure cost.

## E2E: stub or tag-gate, never free-spending

AI-bearing E2E lanes either stub the transport (deterministic fixtures
through this library's record/replay) or run tag-gated against QA with
no AI budget on the billing path (closedloop: `run-ai-e2e` / `run-idp-e2e`
PR labels; the intake E2E deploys a fake-employee fixture lane). Untagged
CI lanes never reach a live model.

See [harness-policy.md](harness-policy.md) for what the library itself
owns (retry, fallback, budgets) vs. these host duties.
