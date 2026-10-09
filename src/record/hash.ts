/**
 * Request Hashing for Record/Replay
 *
 * Stable hashing of LlmRequest instances for record/replay transport lookup.
 * Uses canonical JSON (recursively sorted keys) and SHA-256 via Web Crypto API.
 *
 * @module record/hash
 */

import type { LlmRequest } from '../transport/types';
import { assertJsonSafe } from '../json/jsonSafe';

// ============================================================================
// Tunables
// ============================================================================

/**
 * Default recording format version.
 *
 * Bumped when the recording schema or hash algorithm changes incompatibly.
 * Old recordings must be rejected, not silently migrated.
 *
 * v2 (0.3.0): `temperature` removed from the hashed field set (it is no
 * longer transmitted). Bumped so v1 recordings — whose hashes could depend on
 * temperature — are rejected instead of parsing clean and then silently
 * missing in strict replay (or live-calling in passthrough lanes).
 */
export const DEFAULT_REPLAY_RECORDING_VERSION = 2;

// ============================================================================
// Type Aliases
// ============================================================================

/**
 * Typed failure result from JSON operations.
 *
 * Used instead of throwing errors; callers receive structured diagnostics.
 */
export type JsonSafeFailure = {
  readonly ok: false;
  readonly failure: {
    readonly kind: 'NON_JSON_SAFE';
    readonly reason: string;
  };
};

/**
 * Typed failure result from hashing operations.
 *
 * Used instead of throwing errors; callers receive structured diagnostics.
 */
export type HashFailure = {
  readonly ok: false;
  readonly failure: {
    readonly kind: 'HASH_FAILURE';
    readonly reason: string;
  };
};

// ============================================================================
// Canonical JSON Serialization
// ============================================================================

/**
 * Returns `value` if it is a plain (non-null, non-array) object.
 */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Deep-copies a JSON value with every nested object's keys sorted
 * alphabetically. Arrays keep element order (order is semantic); only object
 * key insertion order is normalized.
 *
 * Non-JSON-safe values (NaN, Infinity, functions, symbols, bigints, circular)
 * are rejected via typed failure, never thrown.
 *
 * @param value - JSON-safe value to canonicalize
 * @returns Canonicalized copy of the value or typed failure
 */
function canonicalizeValue(
  value: unknown
): { readonly ok: true; readonly value: unknown } | JsonSafeFailure {
  // Check JSON safety first (handles circular, NaN, Infinity, functions, symbols, bigints)
  // Use assertJsonSafe to get detailed error messages, convert to typed failure
  try {
    assertJsonSafe(value);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown JSON safety violation';
    return {
      ok: false,
      failure: {
        kind: 'NON_JSON_SAFE',
        reason: message
      }
    };
  }

  if (Array.isArray(value)) {
    const result: unknown[] = [];
    for (const item of value) {
      const itemResult = canonicalizeValue(item);
      if (!itemResult.ok) {
        return itemResult;
      }
      result.push(itemResult.value);
    }
    return { ok: true, value: result };
  }

  if (isPlainObject(value)) {
    const sortedKeys = Object.keys(value).sort();
    const result: Record<string, unknown> = {};
    for (const key of sortedKeys) {
      const keyValue = canonicalizeValue(value[key]);
      if (!keyValue.ok) {
        return keyValue;
      }
      result[key] = keyValue.value;
    }
    return { ok: true, value: result };
  }

  // Primitive values (string, number, boolean, null) are returned as-is
  return { ok: true, value };
}

/**
 * Serializes a value to canonical JSON with recursively sorted object keys.
 *
 * Arrays preserve element order; only object key insertion order is normalized.
 * Non-JSON-safe input returns a typed failure, never throws.
 *
 * @param value - Value to serialize
 * @returns Canonical JSON string or typed failure
 */
export function canonicalJsonStringify(
  value: unknown
): { readonly ok: true; readonly value: string } | JsonSafeFailure {
  const canonicalized = canonicalizeValue(value);
  if (!canonicalized.ok) {
    return canonicalized;
  }

  try {
    return { ok: true, value: JSON.stringify(canonicalized.value) };
  } catch (err: unknown) {
    return {
      ok: false,
      failure: {
        kind: 'NON_JSON_SAFE',
        reason: `JSON.stringify failed: ${err instanceof Error ? err.message : String(err)}`
      }
    };
  }
}

// ============================================================================
// Request Hashing
// ============================================================================

/**
 * Computes a stable SHA-256 hash of an LlmRequest for record/replay lookup.
 *
 * The hash covers the full request: model, systemInstruction, promptText,
 * toolDeclarations, and present sampling parameters (maxOutputTokens,
 * thinkingLevel). Absent optional fields are not included (they use defaults).
 * The deprecated `temperature` field is deliberately NOT hashed: it is never
 * transmitted, so it cannot affect the served response (see v2 bump on
 * DEFAULT_REPLAY_RECORDING_VERSION).
 *
 * Async signature rationale: Web Crypto API (`globalThis.crypto.subtle.digest`)
 * is Promise-based in both browsers and Node 20+. This library must run in both
 * environments without node-only imports.
 *
 * @param req - LlmRequest to hash
 * @returns SHA-256 hex digest or typed failure
 */
export async function inputHashOf(
  req: LlmRequest
): Promise<{ readonly ok: true; readonly value: string } | HashFailure> {
  // Extract only the fields that affect the hash (present optional fields)
  const hashInput: Record<string, unknown> = {
    model: req.model,
    systemInstruction: req.systemInstruction,
    promptText: req.promptText,
    toolDeclarations: req.toolDeclarations
  };

  if (req.maxOutputTokens !== undefined) {
    hashInput.maxOutputTokens = req.maxOutputTokens;
  }
  if (req.thinkingLevel !== undefined) {
    hashInput.thinkingLevel = req.thinkingLevel;
  }

  // Include inlineData in hash for vision inputs (Bug #2 fix)
  if (req.inlineData !== undefined) {
    hashInput.inlineData = req.inlineData;
  }

  const canonicalResult = canonicalJsonStringify(hashInput);
  if (!canonicalResult.ok) {
    return {
      ok: false,
      failure: {
        kind: 'HASH_FAILURE',
        reason: `Failed to canonicalize request: ${canonicalResult.failure.reason}`
      }
    };
  }

  const encoder = new TextEncoder();
  const data = encoder.encode(canonicalResult.value);

  try {
    const hashBuffer = await globalThis.crypto.subtle.digest('SHA-256', data);
    const hashArray = new Uint8Array(hashBuffer);
    const hashHex = Array.from(hashArray)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
    return { ok: true, value: hashHex };
  } catch (err: unknown) {
    return {
      ok: false,
      failure: {
        kind: 'HASH_FAILURE',
        reason: `SHA-256 digest failed: ${err instanceof Error ? err.message : String(err)}`
      }
    };
  }
}
