/**
 * Recommended Model Chain (SSOT)
 *
 * The one place the recommended Gemini Flash fallback chain is declared.
 * App seams import this export — never re-declare a copy — so a model
 * retirement only lands here. Docs under `docs/lessons/model-config.md`
 * reference this export as the source of truth.
 *
 * @module policy/modelChain
 */

/**
 * Recommended fallback chain of Gemini Flash models, attempted in order.
 *
 * Rationale (2026-10):
 * - `gemini-3.8-flash` is pinned as primary: GA, and measurably reduced
 *   loop-spiraling compared to the 3.x models below it in the chain.
 * - 3.5 and 3.6 are dropped: superseded by 3.8 with no residual
 *   availability advantage worth a fallback slot.
 * - `gemini-flash-latest` is terminal-only: an alias that tracks the newest
 *   Flash GA, kept as the last resort so the chain degrades to "whatever
 *   Google currently ships" instead of hard-failing during rollout gaps.
 *   Aliases can change behavior without notice, which is exactly why it must
 *   never sit above a pinned version.
 */
export const RECOMMENDED_FLASH_MODEL_CHAIN = [
  'gemini-3.8-flash',
  'gemini-3.7-flash',
  'gemini-flash-latest'
] as const;

/** Element type of {@link RECOMMENDED_FLASH_MODEL_CHAIN}. */
export type RecommendedFlashModel = (typeof RECOMMENDED_FLASH_MODEL_CHAIN)[number];
