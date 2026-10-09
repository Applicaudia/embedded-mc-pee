/**
 * Behavior tests for {@link createGeminiTransport} — the Gemini LLM transport.
 *
 * Tests verify configuration handling, request mapping, response extraction,
 * timeout behavior, and ADR-0001 compliance (no `responseSchema` ever).
 * Uses a stubbed `GoogleGenAI` client to avoid live network calls.
 */

import { describe, expect, it, vi } from 'vitest';
import type { ToolContract } from '../tool/toolContract';
import { setDebug } from '../debug/debug';
import {
  createGeminiTransport,
  DEFAULT_GEMINI_TIMEOUT_MS
} from './geminiTransport';

// Enable debug mode for tests that assert dbg side effects
setDebug(true);

// ============================================================================
// Stub types and helpers
// ============================================================================

/**
 * Stub implementation of `@google/genai` types needed for testing.
 *
 * Only the fields we actually use in the transport are stubbed; unknown
 * fields are ignored. This keeps tests decoupled from SDK changes to unrelated
 * parts of the type.
 */
interface StubGenerateContentResult {
  candidates?: Array<{
    finishReason?: string;
    content?: {
      parts?: Array<
        | { text: string }
        | { functionCall: { name: string; args: unknown } }
      >;
    };
  }>;
  usageMetadata?: {
    candidatesTokenCount?: number;
    thoughtsTokenCount?: number;
  };
}

/**
 * Minimal stub for the `@google/genai` client.
 *
 * Only the `models.generateContent` method is used by the transport; other
 * SDK surface is omitted to keep tests focused.
 */
interface StubGoogleGenAI {
  models: {
    generateContent: ReturnType<typeof vi.fn>;
  };
}

/**
 * Creates a mock `GoogleGenAI` client stub.
 *
 * @param resolveWith - Response the stub should resolve with (or undefined for error tests)
 * @param rejectWithError - Error the stub should reject with (takes precedence over resolveWith)
 * @param delayMs - Artificial delay before resolve/reject (for timeout tests)
 */
function createMockClient(
  resolveWith?: StubGenerateContentResult,
  rejectWithError?: Error,
  delayMs: number = 0
): StubGoogleGenAI {
  return {
    models: {
      generateContent: vi.fn(
        async (
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          _config: {
            model: string;
            contents: Array<{ role: string; parts: Array<{ text: string }> }>;
            config?: Record<string, unknown>;
          }
        ) => {
          if (delayMs > 0) {
            await new Promise(resolve => setTimeout(resolve, delayMs));
          }

          if (rejectWithError) {
            throw rejectWithError;
          }

          if (resolveWith) {
            return resolveWith as unknown;
          }

          // Default minimal response
          return {
            candidates: [
              {
                finishReason: 'STOP',
                content: {
                  parts: [{ text: '{"result": "ok"}' }]
                }
              }
            ]
          } as unknown;
        }
      )
    }
  };
}

/**
 * Helper to deep-scan a value for `responseSchema` in any nested object/array.
 *
 * Used for ADR-0001 enforcement tests ensuring the transport NEVER sends
 * `responseSchema` to the SDK.
 */
function containsResponseSchema(value: unknown): boolean {
  if (value === null || typeof value !== 'object') {
    return false;
  }

  if (Array.isArray(value)) {
    return value.some(item => containsResponseSchema(item));
  }

  const obj = value as Record<string, unknown>;
  if ('responseSchema' in obj) {
    return true;
  }

  return Object.values(obj).some(v => containsResponseSchema(v));
}

/**
 * Helper tool contract for tests.
 */
const testTool: ToolContract = {
  name: 'test_tool',
  description: 'A test tool',
  inputSchema: {
    type: 'object',
    properties: {
      message: { type: 'string' }
    },
    required: ['message']
  }
};

// ============================================================================
// Test suite
// ============================================================================

describe('createGeminiTransport', () => {
  describe('configuration validation', () => {
    it('fails when neither apiKey nor client is provided', () => {
      const result = createGeminiTransport({});

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.failures).toContain(
          'Either apiKey or client must be provided (transport needs a way to authenticate to Gemini)'
        );
      }
    });

    it('fails when both apiKey and client are provided', () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        apiKey: 'test-key',
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.failures).toContain(
          'Only one of apiKey or client may be provided (both are present)'
        );
      }
    });

    it('accepts apiKey only', () => {
      const result = createGeminiTransport({ apiKey: 'test-key' });

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.transport).toBeDefined();
        expect(result.transport.complete).toBeInstanceOf(Function);
      }
    });

    it('accepts client only', () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.transport).toBeDefined();
        expect(result.transport.complete).toBeInstanceOf(Function);
      }
    });

    it('uses DEFAULT_GEMINI_TIMEOUT_MS when no timeout is provided', () => {
      const result = createGeminiTransport({ apiKey: 'test-key' });
      expect(result.ok).toBe(true);
      // Timeout is internal; we verify it indirectly via timeout behavior tests
      expect(DEFAULT_GEMINI_TIMEOUT_MS).toBe(15_000);
    });
  });

  describe('request mapping', () => {
    it('sends systemInstruction and promptText as contents', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'You are a helpful assistant.',
          promptText: 'Hello, world!',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
      expect(generateContentSpy).toHaveBeenCalled();

      const capturedCall = generateContentSpy.mock.calls[0]?.[0] as {
        model: string;
        contents: unknown;
        config?: Record<string, unknown>;
      } | undefined;
      expect(capturedCall).toBeDefined();
      expect(capturedCall?.config?.systemInstruction).toBe('You are a helpful assistant.');
      expect(capturedCall?.contents).toEqual([
        { role: 'user', parts: [{ text: 'Hello, world!' }] }
      ]);
    });

    it('prepends inlineData part for vision inputs', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'You transcribe documents.',
          promptText: 'Transcribe this document.',
          toolDeclarations: [],
          inlineData: { mimeType: 'application/pdf', data: 'aGVsbG8=' }
        },
        { timeoutMs: 10_000 }
      );

      const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
      const capturedCall = generateContentSpy.mock.calls[0]?.[0] as {
        contents: unknown;
      } | undefined;
      expect(capturedCall?.contents).toEqual([
        {
          role: 'user',
          parts: [
            { inlineData: { mimeType: 'application/pdf', data: 'aGVsbG8=' } },
            { text: 'Transcribe this document.' }
          ]
        }
      ]);
    });

    it('includes functionDeclarations when tools are present', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'You are a helpful assistant.',
          promptText: 'Use the tool.',
          toolDeclarations: [testTool]
        },
        { timeoutMs: 10_000 }
      );

      const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
      const capturedConfig = generateContentSpy.mock.calls[0]?.[0]?.config as Record<string, unknown>;

      expect(capturedConfig?.tools).toBeDefined();
      expect(Array.isArray(capturedConfig?.tools)).toBe(true);
      expect((capturedConfig.tools as Array<unknown>)[0]).toMatchObject({
        functionDeclarations: [
          {
            name: 'test_tool',
            description: 'A test tool',
            parameters: {
              type: 'object',
              properties: {
                message: { type: 'string' }
              },
              required: ['message']
            }
          }
        ]
      });
    });

    it('omits tools key entirely when there are no tool declarations', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'You are a helpful assistant.',
          promptText: 'Hello!',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
      const capturedConfig = generateContentSpy.mock.calls[0]?.[0]?.config as Record<string, unknown>;

      // Key should not exist at all (not undefined, not empty array)
      expect('tools' in capturedConfig).toBe(false);
    });

    it('NEVER sets responseSchema (ADR-0001 compliance)', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      // Test with various request shapes
      const testCases = [
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: [testTool]
        }
      ];

      for (const req of testCases) {
        await result.transport.complete(req, { timeoutMs: 10_000 });

        const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
        const capturedConfig = generateContentSpy.mock.calls.at(-1)?.[0]?.config as Record<string, unknown>;

        // Deep-scan the entire config structure for responseSchema
        const configJson = JSON.stringify(capturedConfig);
        expect(configJson).not.toContain('responseSchema');

        // Also check via recursive scan
        expect(containsResponseSchema(capturedConfig)).toBe(false);
      }
    });

    it('NEVER transmits temperature; passes through maxOutputTokens and thinkingLevel', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: [],
          // Deprecated since 0.3.0: upcoming Gemini models error on temperature.
          // Present on the request, must be dropped by the transport.
          temperature: 0.7,
          maxOutputTokens: 4096,
          thinkingLevel: 'low'
        },
        { timeoutMs: 10_000 }
      );

      const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
      const capturedConfig = generateContentSpy.mock.calls[0]?.[0]?.config as Record<string, unknown>;

      expect('temperature' in (capturedConfig ?? {})).toBe(false);
      expect(capturedConfig?.maxOutputTokens).toBe(4096);
      expect(capturedConfig?.thinkingConfig).toEqual({ thinkingLevel: 'low' });
    });

    it('passes through medium/high thinkingLevel without a cast (post-minimal union)', async () => {
      const mockClient = createMockClient();
      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      for (const thinkingLevel of ['medium', 'high'] as const) {
        await result.transport.complete(
          {
            model: 'gemini-test',
            systemInstruction: 'Test',
            promptText: 'Test',
            toolDeclarations: [],
            thinkingLevel
          },
          { timeoutMs: 10_000 }
        );
      }

      const generateContentSpy = mockClient.models.generateContent as ReturnType<typeof vi.fn>;
      const firstConfig = generateContentSpy.mock.calls[0]?.[0]?.config as Record<string, unknown>;
      const secondConfig = generateContentSpy.mock.calls[1]?.[0]?.config as Record<string, unknown>;
      expect(firstConfig?.thinkingConfig).toEqual({ thinkingLevel: 'medium' });
      expect(secondConfig?.thinkingConfig).toEqual({ thinkingLevel: 'high' });
    });

    it('uses the smaller of transport timeout and per-call timeout', async () => {
      // This is tested via timeout behavior below
      const slowMockClient = createMockClient(undefined, undefined, 20_000); // 20s delay
      const result = createGeminiTransport({
        client: slowMockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client'],
        timeoutMs: 5_000 // 5s transport timeout
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      // Per-call timeout of 1s should be used (smaller than transport's 5s)
      await expect(
        result.transport.complete(
          {
            model: 'gemini-test',
            systemInstruction: 'Test',
            promptText: 'Test',
            toolDeclarations: []
          },
          { timeoutMs: 1_000 } // 1s per-call timeout
        )
      ).rejects.toThrow(/timeout/);
    });
  });

  describe('response extraction', () => {
    it('extracts rawText from text-only response', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [{ text: 'Plain text response' }]
            }
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.rawText).toBe('Plain text response');
      expect(response.toolCall).toBeUndefined();
      expect(response.finishReason).toBe('STOP');
    });

    it('joins multiple text parts', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [
                { text: 'First part' },
                { text: ' second part' },
                { text: ' third part' }
              ]
            }
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.rawText).toBe('First part second part third part');
    });

    it('extracts toolCall from function call response', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [
                {
                  functionCall: {
                    name: 'test_tool',
                    args: { message: 'hello' }
                  }
                }
              ]
            }
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: [testTool]
        },
        { timeoutMs: 10_000 }
      );

      expect(response.toolCall).toEqual({
        name: 'test_tool',
        args: { message: 'hello' }
      });
    });

    it('extracts first toolCall when multiple function calls are present', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [
                { functionCall: { name: 'tool_one', args: { a: 1 } } },
                { functionCall: { name: 'tool_two', args: { b: 2 } } }
              ]
            }
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: [testTool]
        },
        { timeoutMs: 10_000 }
      );

      // Should have extracted the first call only
      expect(response.toolCall).toEqual({
        name: 'tool_one',
        args: { a: 1 }
      });
      // And logged a warning about the extra call
      // (we can't easily assert the warning without capturing console output)
    });

    it('returns args: undefined for malformed function call args', async () => {
      const testCases = [
        { name: 'missing args', args: undefined },
        { name: 'null args', args: null },
        { name: 'array args', args: [1, 2, 3] },
        { name: 'string args', args: 'invalid' }
      ];

      for (const { args } of testCases) {
        const mockClient = createMockClient({
          candidates: [
            {
              finishReason: 'STOP',
              content: {
                parts: [
                  {
                    functionCall: {
                      name: 'test_tool',
                      args: args as unknown
                    }
                  }
                ]
              }
            }
          ]
        });

        const result = createGeminiTransport({
          client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
        });

        expect(result.ok).toBe(true);
        if (!result.ok) return;

        const response = await result.transport.complete(
          {
            model: 'gemini-test',
            systemInstruction: 'Test',
            promptText: 'Test',
            toolDeclarations: [testTool]
          },
          { timeoutMs: 10_000 }
        );

        expect(response.toolCall).toEqual({
          name: 'test_tool',
          args: undefined
        });
        // Warning logged (can't assert easily without console capture)
      }
    });

    it('returns empty string for response with no text and no tool call', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: []
            }
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.rawText).toBe('');
      expect(response.toolCall).toBeUndefined();
    });

    it('propagates finishReason verbatim', async () => {
      const finishReasons = ['STOP', 'MAX_TOKENS', 'RECITATION', 'SAFETY'];

      for (const reason of finishReasons) {
        const mockClient = createMockClient({
          candidates: [
            {
              finishReason: reason,
              content: {
                parts: [{ text: 'test' }]
              }
            }
          ]
        });

        const result = createGeminiTransport({
          client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
        });

        expect(result.ok).toBe(true);
        if (!result.ok) return;

        const response = await result.transport.complete(
          {
            model: 'gemini-test',
            systemInstruction: 'Test',
            promptText: 'Test',
            toolDeclarations: []
          },
          { timeoutMs: 10_000 }
        );

        expect(response.finishReason).toBe(reason);
      }
    });

    it('maps usage metadata when present', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [{ text: 'test' }]
            }
          }
        ],
        usageMetadata: {
          candidatesTokenCount: 1234,
          thoughtsTokenCount: 567
        }
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.usage).toEqual({
        outputTokens: 1234,
        thoughtsTokenCount: 567
      });
    });

    it('omits usage when not present in response', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              parts: [{ text: 'test' }]
            }
          }
        ]
        // No usageMetadata
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.usage).toBeUndefined();
    });
  });

  describe('error handling', () => {
    it('rethrows SDK errors unchanged', async () => {
      const sdkError = new Error('SDK network failure');
      const mockClient = createMockClient(undefined, sdkError);

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      await expect(
        result.transport.complete(
          {
            model: 'gemini-test',
            systemInstruction: 'Test',
            promptText: 'Test',
            toolDeclarations: []
          },
          { timeoutMs: 10_000 }
        )
      ).rejects.toThrow(sdkError);
    });

    it('times out when SDK call takes too long', async () => {
      // Use vi.useFakeTimers for timeout testing
      vi.useFakeTimers();

      const neverResolvingClient = createMockClient(
        undefined,
        undefined,
        100_000 // Very long delay, timeout should fire first
      );

      const result = createGeminiTransport({
        client: neverResolvingClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const timeoutPromise = result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 500 } // Short timeout
      );

      // Attach the rejection handler BEFORE advancing timers, so the rejection
      // is never momentarily unhandled when the timer fires
      const rejection = expect(timeoutPromise).rejects.toThrow(/timeout/);

      // Advance time past the timeout
      await vi.advanceTimersByTimeAsync(500);

      await rejection;

      vi.useRealTimers();
    });

    it('includes timeout in error message', async () => {
      vi.useFakeTimers();

      const neverResolvingClient = createMockClient(undefined, undefined, 100_000);
      const result = createGeminiTransport({
        client: neverResolvingClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const timeoutPromise = result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 100 }
      );

      // Attach the rejection handler BEFORE advancing timers, so the rejection
      // is never momentarily unhandled when the timer fires
      const rejection = expect(timeoutPromise).rejects.toThrow(/timeout/);

      await vi.advanceTimersByTimeAsync(100);

      await rejection;

      vi.useRealTimers();
    });
  });

  describe('edge cases', () => {
    it('handles response with undefined candidates', async () => {
      const mockClient = createMockClient({
        // No candidates field
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      // Should handle gracefully with empty text
      expect(response.rawText).toBe('');
      expect(response.toolCall).toBeUndefined();
      expect(response.finishReason).toBeUndefined();
    });

    it('handles response with empty candidates array', async () => {
      const mockClient = createMockClient({
        candidates: []
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.rawText).toBe('');
      expect(response.toolCall).toBeUndefined();
      expect(response.finishReason).toBeUndefined();
    });

    it('handles candidate with undefined content', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP'
            // No content field
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.rawText).toBe('');
      expect(response.finishReason).toBe('STOP');
    });

    it('handles content with undefined parts', async () => {
      const mockClient = createMockClient({
        candidates: [
          {
            finishReason: 'STOP',
            content: {
              // No parts field
            }
          }
        ]
      });

      const result = createGeminiTransport({
        client: mockClient as unknown as Parameters<typeof createGeminiTransport>[0]['client']
      });

      expect(result.ok).toBe(true);
      if (!result.ok) return;

      const response = await result.transport.complete(
        {
          model: 'gemini-test',
          systemInstruction: 'Test',
          promptText: 'Test',
          toolDeclarations: []
        },
        { timeoutMs: 10_000 }
      );

      expect(response.rawText).toBe('');
      expect(response.finishReason).toBe('STOP');
    });
  });
});
