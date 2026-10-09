/**
 * Tests for record/hash.ts
 *
 * Hash stability, canonical JSON serialization, and edge cases.
 */

import { describe, it, expect } from 'vitest';
import { canonicalJsonStringify, inputHashOf, DEFAULT_REPLAY_RECORDING_VERSION } from './hash';
import type { LlmRequest } from '../transport/types';

describe('canonicalJsonStringify', () => {
  it('should return primitive values unchanged', () => {
    const result = canonicalJsonStringify('hello');
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('"hello"');
    }
  });

  it('should return numbers unchanged', () => {
    const result = canonicalJsonStringify(42);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('42');
    }
  });

  it('should return booleans unchanged', () => {
    const result = canonicalJsonStringify(true);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('true');
    }
  });

  it('should return null unchanged', () => {
    const result = canonicalJsonStringify(null);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('null');
    }
  });

  it('should sort object keys alphabetically', () => {
    const obj = { z: 1, a: 2, m: 3 };
    const result = canonicalJsonStringify(obj);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('{"a":2,"m":3,"z":1}');
    }
  });

  it('should sort nested object keys recursively', () => {
    const obj = { outer2: { z: 1, a: 2 }, outer1: { m: 3, b: 4 } };
    const result = canonicalJsonStringify(obj);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('{"outer1":{"b":4,"m":3},"outer2":{"a":2,"z":1}}');
    }
  });

  it('should preserve array element order', () => {
    const arr = [3, 1, 2];
    const result = canonicalJsonStringify(arr);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('[3,1,2]');
    }
  });

  it('should sort object keys inside arrays', () => {
    const arr = [{ z: 1, a: 2 }, { m: 3, b: 4 }];
    const result = canonicalJsonStringify(arr);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe('[{"a":2,"z":1},{"b":4,"m":3}]');
    }
  });

  it('should reject NaN with typed failure', () => {
    const result = canonicalJsonStringify(NaN);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('non-finite number');
    }
  });

  it('should reject Infinity with typed failure', () => {
    const result = canonicalJsonStringify(Infinity);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('non-finite number');
    }
  });

  it('should reject -Infinity with typed failure', () => {
    const result = canonicalJsonStringify(-Infinity);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('non-finite number');
    }
  });

  it('should reject functions with typed failure', () => {
    const result = canonicalJsonStringify(() => {});
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('unsupported type function');
    }
  });

  it('should reject symbols with typed failure', () => {
    const result = canonicalJsonStringify(Symbol('test'));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('unsupported type symbol');
    }
  });

  it('should reject bigints with typed failure', () => {
    const result = canonicalJsonStringify(BigInt(123));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('unsupported type bigint');
    }
  });

  it('should reject circular objects with typed failure', () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const result = canonicalJsonStringify(circular);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('NON_JSON_SAFE');
      expect(result.failure.reason).toContain('circular reference');
    }
  });
});

describe('inputHashOf', () => {
  const makeRequest = (overrides?: Partial<LlmRequest>): LlmRequest => ({
    model: 'gemini-2.0-flash',
    systemInstruction: 'You are a helpful assistant.',
    promptText: 'What is 2+2?',
    toolDeclarations: [],
    ...overrides
  });

  it('should compute stable hash for identical requests', async () => {
    const req = makeRequest();
    const hash1 = await inputHashOf(req);
    const hash2 = await inputHashOf(req);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      expect(hash1.value).toBe(hash2.value);
      expect(hash1.value).toMatch(/^[a-f0-9]{64}$/);
    }
  });

  it('should produce same hash for semantically identical requests with different key insertion order', async () => {
    const baseReq = makeRequest();

    // Create two objects with same content but different insertion order
    const req1: LlmRequest = {
      model: baseReq.model,
      systemInstruction: baseReq.systemInstruction,
      promptText: baseReq.promptText,
      toolDeclarations: [...baseReq.toolDeclarations],
      temperature: 0.5,
      maxOutputTokens: 1000
    };

    const req2: LlmRequest = {
      maxOutputTokens: 1000,
      temperature: 0.5,
      model: baseReq.model,
      toolDeclarations: [...baseReq.toolDeclarations],
      systemInstruction: baseReq.systemInstruction,
      promptText: baseReq.promptText
    };

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      expect(hash1.value).toBe(hash2.value);
    }
  });

  it('should produce different hashes for different prompts', async () => {
    const req1 = makeRequest({ promptText: 'What is 2+2?' });
    const req2 = makeRequest({ promptText: 'What is 3+3?' });

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      expect(hash1.value).not.toBe(hash2.value);
    }
  });

  it('should produce different hashes for different models', async () => {
    const req1 = makeRequest({ model: 'gemini-2.0-flash' });
    const req2 = makeRequest({ model: 'gemini-2.0-pro' });

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      expect(hash1.value).not.toBe(hash2.value);
    }
  });

  it('should NOT include deprecated temperature in the hash (0.3.0 removal)', async () => {
    const req1 = makeRequest({ temperature: 0.1 });
    const req2 = makeRequest({ temperature: 0.9 });
    const req3 = makeRequest({}); // no temperature

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);
    const hash3 = await inputHashOf(req3);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    expect(hash3.ok).toBe(true);
    if (hash1.ok && hash2.ok && hash3.ok) {
      // Temperature is never transmitted, so it cannot affect the served
      // response and must not split replay recordings.
      expect(hash1.value).toBe(hash3.value);
      expect(hash2.value).toBe(hash3.value);
    }
  });

  it('should produce different hashes for presence vs absence of hashed optional fields', async () => {
    const req1 = makeRequest({ maxOutputTokens: 1000 });
    const req2 = makeRequest({}); // no maxOutputTokens

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      expect(hash1.value).not.toBe(hash2.value);
    }
  });

  it('should reject non-JSON-safe request with typed failure', async () => {
    const req = makeRequest();
    // Inject non-JSON-safe value
    (req as { toolDeclarations: unknown }).toolDeclarations = [
      { name: 'badTool', inputSchema: NaN }
    ];

    const result = await inputHashOf(req);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('HASH_FAILURE');
      expect(result.failure.reason).toContain('Failed to canonicalize');
    }
  });

  // Regression test for bug #2: inlineData missing from replay hash
  it('should produce different hashes for different inlineData content', async () => {
    const req1: LlmRequest = {
      model: 'gemini-2.0-flash',
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'What is in this image?',
      toolDeclarations: [],
      inlineData: {
        mimeType: 'image/png',
        data: 'base64encodedimage1'
      }
    };

    const req2: LlmRequest = {
      model: 'gemini-2.0-flash',
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'What is in this image?',
      toolDeclarations: [],
      inlineData: {
        mimeType: 'image/png',
        data: 'base64encodedimage2'
      }
    };

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      // Different inlineData should produce different hashes
      expect(hash1.value).not.toBe(hash2.value);
    }
  });

  it('should produce same hash for identical inlineData', async () => {
    const baseInlineData = {
      mimeType: 'image/jpeg',
      data: 'abc123def456'
    };

    const req1: LlmRequest = {
      model: 'gemini-2.0-flash',
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'Describe this image',
      toolDeclarations: [],
      inlineData: baseInlineData
    };

    const req2: LlmRequest = {
      model: 'gemini-2.0-flash',
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'Describe this image',
      toolDeclarations: [],
      inlineData: { ...baseInlineData }
    };

    const hash1 = await inputHashOf(req1);
    const hash2 = await inputHashOf(req2);

    expect(hash1.ok).toBe(true);
    expect(hash2.ok).toBe(true);
    if (hash1.ok && hash2.ok) {
      expect(hash1.value).toBe(hash2.value);
    }
  });
});

describe('DEFAULT_REPLAY_RECORDING_VERSION', () => {
  it('should be a positive integer', () => {
    expect(DEFAULT_REPLAY_RECORDING_VERSION).toBeGreaterThan(0);
    expect(Number.isInteger(DEFAULT_REPLAY_RECORDING_VERSION)).toBe(true);
  });

  // Tripwire: v2 removed temperature from the hashed field set. If this
  // bumps again, recordings must be regenerated in the same commit as the
  // hash change and this pin moves with it.
  it('should be pinned to 2 (temperature removal)', () => {
    expect(DEFAULT_REPLAY_RECORDING_VERSION).toBe(2);
  });
});
