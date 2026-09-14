/**
 * Transient Error Detection
 *
 * Classifies provider errors as transient (retryable) vs permanent.
 * Uses duck-typing on error shapes to avoid direct SDK dependencies.
 *
 * Ported from courtpuzzle/functions/src/providers/demandExtractionProvider.ts.
 */

/**
 * Tests whether an error represents a transient provider condition that
 * justifies retry (rate limits, resource exhaustion, temporary unavailability).
 *
 * Uses duck-typing on error shapes to stay vendor-agnostic:
 * - Numeric status codes (429, 503)
 * - String status codes ('RESOURCE_EXHAUSTED', 'UNAVAILABLE')
 * - Embedded JSON in error messages containing these codes
 *
 * @param error - Unknown error value from an SDK or API call
 * @returns true if the error is transient/retryable, false otherwise
 */
export function isTransientProviderError(error: unknown): boolean {
  if (!error) return false;

  const status = (error as { status?: number | string })?.status;
  if (status === 429 || status === '429' || status === 'RESOURCE_EXHAUSTED') return true;
  if (status === 503 || status === '503' || status === 'UNAVAILABLE') return true;

  const code = (error as { code?: number | string })?.code;
  if (code === 429 || code === '429' || code === 'RESOURCE_EXHAUSTED') return true;
  if (code === 503 || code === '503' || code === 'UNAVAILABLE') return true;
  if (code === 'ECONNRESET' || code === 'ETIMEDOUT' || code === 'ENOTFOUND' || code === 'UND_ERR_CONNECT_TIMEOUT') return true;

  const message = (error as Error)?.message ?? String(error);
  if (/\b(429|503)\b/.test(message)) return true;
  if (/RESOURCE_EXHAUSTED|UNAVAILABLE|overloaded|rate\s*limit|quota\s*exceeded|too\s*many\s*requests|fetch\s*failed|network\s*error/i.test(message)) return true;

  return false;
}
