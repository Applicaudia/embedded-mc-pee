/**
 * Gemini LLM Transport Implementation
 *
 * A thin adapter that implements {@link LlmTransport} for the Google Gemini
 * API via the `@google/genai` SDK. The transport is a factory function
 * (`createGeminiTransport`) that returns either a configured transport or a
 * typed config failure — never throws for configuration errors.
 *
 * Per ADR-0001, this transport NEVER sets `responseSchema`. The schema-in-prompt
 * contract lives in the system instruction and prompt text; runtime validation
 * enforces the shape. This prevents the token-repetition loops that Gemini 3.x
 * Flash models exhibit under large responseSchema constraints.
 *
 * Transport duties (per docs/mcp-contract-mapping.md):
 * - Map tool declarations to functionDeclarations
 * - Send systemInstruction and promptText as contents
 * - Apply timeout as a backstop (the harness also races)
 * - Extract rawText and/or toolCall from responses
 * - Propagate finishReason and usage metadata verbatim
 * - Rethrow SDK errors (the harness classifies them as transient/permanent)
 *
 * @module
 */

import { GoogleGenAI } from '@google/genai';
import type {
  GenerateContentResponse,
  Content,
  GenerateContentConfig,
  Part,
  FunctionDeclaration,
  FunctionCall,
  ThinkingConfig
} from '@google/genai';
import { dbg } from '../debug/debug';
import type { LlmTransport, LlmRequest, LlmResponse } from '../transport/types';
import { functionDeclarationOf } from './schemaToFunctionDeclaration';

// ============================================================================
// Configuration defaults
// ============================================================================

/**
 * Default per-call timeout in milliseconds.
 *
 * The transport applies its own timeout race as a backstop even though the
 * harness also races. This double-timeout approach ensures the call fails
 * fast even if the harness's timeout mechanism fails or is misconfigured.
 * 15 seconds is generous for Gemini Flash models; longer timeouts delay
 * fallback model activation.
 */
export const DEFAULT_GEMINI_TIMEOUT_MS = 15_000;

// ============================================================================
// Options and error types
// ============================================================================

/**
 * Configuration options for the Gemini transport factory.
 *
 * All tunables have exported `DEFAULT_*` constants and are optional. Exactly
 * one of `apiKey` or `client` must be provided; the factory returns a typed
 * failure rather than throwing when neither or both are supplied.
 */
export interface GeminiTransportOptions {
  /**
   * Google Gemini API key. Optional; if not provided, the transport attempts
   * to read from the `GEMINI_API_KEY` environment variable. Exactly one of
   * `apiKey` or `client` must be specified.
   */
  readonly apiKey?: string;

  /**
   * Pre-configured `@google/genai` client. Optional; useful for tests and
   * hosts that need custom SDK configuration. Exactly one of `apiKey` or
   * `client` must be specified.
   */
  readonly client?: GoogleGenAI;

  /**
   * Per-call timeout in milliseconds. Defaults to
   * {@link DEFAULT_GEMINI_TIMEOUT_MS}. The transport races the SDK promise
   * against a timer; timeout → reject with an Error whose message includes
   * 'timeout' (the harness classifies this as non-transient).
   */
  readonly timeoutMs?: number;
}

/**
 * Discriminated union result type for the transport factory.
 *
 * Success: `{ok: true, transport}` — a working `LlmTransport` implementation.
 * Failure: `{ok: false, failures}` — array of config error messages.
 *
 * This shape follows the library convention: factory functions return typed
 * results rather than throwing for configuration errors.
 */
export type GeminiTransportResult =
  | { readonly ok: true; readonly transport: LlmTransport }
  | { readonly ok: false; readonly failures: readonly string[] };

/**
 * Typed error class for Gemini transport configuration failures.
 *
 * The factory uses this internally to aggregate configuration problems before
 * wrapping them in a `GeminiTransportResult`. Hosts can check the `failures`
 * array for specific problems without catching exceptions.
 */
export class GeminiTransportConfigError extends Error {
  /**
   * Array of configuration failure messages.
   */
  public readonly failures: readonly string[];

  constructor(message: string, failures: readonly string[]) {
    super(message);
    this.name = 'GeminiTransportConfigError';
    this.failures = failures;
  }
}

// ============================================================================
// Internal helpers
// ============================================================================

/**
 * Races a promise against a timeout. Rejects with an Error whose message
 * includes 'timeout' if the timer fires first.
 *
 * @param operation - The promise to race
 * @param timeoutMs - Timeout in milliseconds
 * @param timeoutMessage - Message for the timeout Error
 * @returns The operation result, or rejects on timeout/operation failure
 */
function rejectAfterTimeout<T>(
  operation: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      const timeoutError = new Error(timeoutMessage);
      // Ensure the error message contains 'timeout' for harness classification
      if (!timeoutError.message.toLowerCase().includes('timeout')) {
        timeoutError.message = `${timeoutMessage} (timeout)`;
      }
      reject(timeoutError);
    }, timeoutMs);

    operation.then(
      result => {
        clearTimeout(timeoutId);
        resolve(result);
      },
      error => {
        clearTimeout(timeoutId);
        reject(error);
      }
    );
  });
}

/**
 * Joins text parts from a Gemini response. Handles both single-text and
 * multi-part responses.
 *
 * @param parts - Array of parts from a response candidate
 * @returns Joined text content, or empty string if no text parts exist
 */
function joinTextParts(parts: readonly Part[] | undefined): string {
  if (!parts || parts.length === 0) {
    return '';
  }

  const textParts: string[] = [];
  for (const part of parts) {
    if (part.text !== undefined) {
      textParts.push(part.text);
    }
  }

  return textParts.join('');
}

/**
 * Extracts the first function call from a response, logging a warning if
 * multiple calls are present (the harness decides precedence; the transport
 * does not).
 *
 * @param functionCalls - Array of function calls from the response
 * @returns The first call, or undefined if no calls
 */
function extractFirstFunctionCall(
  functionCalls: readonly FunctionCall[] | undefined
): { name: string; args: unknown } | undefined {
  if (!functionCalls || functionCalls.length === 0) {
    return undefined;
  }

  if (functionCalls.length > 1) {
    dbg.warn(
      `[GeminiTransport] Multiple function calls in response; taking first and ignoring ${functionCalls.length - 1} extras`
    );
  }

  const first = functionCalls[0];
  const name = first.name;
  if (!name) {
    return undefined;
  }

  const args = first.args;

  // Log a warning if args are malformed (missing, null, or not a plain object)
  // NOTE: typeof null === 'object' in JavaScript, so we must check explicitly
  if (args === undefined || args === null || typeof args !== 'object' || Array.isArray(args)) {
    dbg.warn('[GeminiTransport] Function call has malformed args; passing undefined', {
      name,
      args
    });
    return { name, args: undefined };
  }

  return { name, args };
}

// ============================================================================
// Transport factory
// ============================================================================

/**
 * Creates a Gemini LLM transport with the given options.
 *
 * Exactly one of `apiKey` or `client` must be provided. If neither or both are
 * supplied, returns a typed failure rather than throwing. This factory pattern
 * allows hosts to detect configuration problems at startup without try/catch.
 *
 * The transport:
 * - Maps `toolDeclarations` to `functionDeclarations` (omits `tools` key when
 *   there are no tools — an empty array is invalid in the SDK).
 * - Sends `systemInstruction`, `contents`, and the sampling parameters that
 *   survive 0.3.0 (`maxOutputTokens`, `thinkingLevel`); temperature is never
 *   sent (upcoming Gemini models error on it).
 * - NEVER sets `responseSchema` per ADR-0001 (prevents repetition loops).
 * - Races the SDK call against a timeout (backstop to the harness's timeout).
 * - Extracts `rawText` and/or `toolCall` from responses.
 * - Propagates `finishReason` and `usage` verbatim.
 * - Rethrows SDK errors unchanged (the harness classifies them).
 *
 * @param options - Configuration options
 * @returns {@link GeminiTransportResult} — either a working transport or config failures
 */
export function createGeminiTransport(
  options: GeminiTransportOptions
): GeminiTransportResult {
  const failures: string[] = [];

  // Validate: exactly one of apiKey or client must be provided
  const hasApiKey = options.apiKey !== undefined;
  const hasClient = options.client !== undefined;

  if (!hasApiKey && !hasClient) {
    failures.push(
      'Either apiKey or client must be provided (transport needs a way to authenticate to Gemini)'
    );
  } else if (hasApiKey && hasClient) {
    failures.push(
      'Only one of apiKey or client may be provided (both are present)'
    );
  }

  if (failures.length > 0) {
    return { ok: false, failures };
  }

  // Resolve the client
  let client: GoogleGenAI;
  if (hasClient) {
    // hasClient is true here, so client is defined
    client = options.client as GoogleGenAI;
  } else {
    // apiKey is present here (validated above)
    const apiKey = options.apiKey as string;
    try {
      client = new GoogleGenAI({ apiKey });
    } catch {
      failures.push('Failed to construct GoogleGenAI client from apiKey');
      return { ok: false, failures };
    }
  }

  // Build the transport implementation
  const transport: LlmTransport = {
    complete: async (req: LlmRequest, opts: { timeoutMs: number }): Promise<LlmResponse> => {
      const effectiveTimeoutMs = options.timeoutMs !== undefined
        ? Math.min(options.timeoutMs, opts.timeoutMs)
        : opts.timeoutMs;

      // Map tool declarations to function declarations
      const functionDeclarations: FunctionDeclaration[] = req.toolDeclarations.map(
        functionDeclarationOf
      );

      // Build the config
      const config: GenerateContentConfig = {
        systemInstruction: req.systemInstruction
      };

      // Add sampling parameters when present. Temperature is deliberately
      // NEVER forwarded: upcoming Gemini models error on temperature/top_p/
      // top_k. (The deprecated LlmRequest.temperature field is ignored here.)
      if (req.maxOutputTokens !== undefined) {
        config.maxOutputTokens = req.maxOutputTokens;
      }
      if (req.thinkingLevel !== undefined) {
        // The REST wire format is the lowercase level names ("low" |
        // "medium" | "high" — ai.google.dev generate-content ThinkingConfig;
        // the SDK passes thinkingConfig through verbatim, and the lowercase
        // form is production-verified). The SDK's exported `ThinkingLevel`
        // enum uses UPPERCASE member names, so the value is widened to the
        // declared field type instead of being cast to the enum: the runtime
        // string is the contract, not the enum.
        config.thinkingConfig = {
          thinkingLevel: req.thinkingLevel
        } as unknown as ThinkingConfig;
      }

      // Build contents. Vision inputs (inlineData) come first, followed by
      // the prompt text; text-only requests are unchanged.
      const parts: Part[] = [];
      if (req.inlineData !== undefined) {
        parts.push({ inlineData: req.inlineData });
      }
      parts.push({ text: req.promptText });
      const contents: Content[] = [
        {
          role: 'user',
          parts
        }
      ];

      // Add tools only when we have function declarations
      // (empty functionDeclarations array is invalid in the SDK)
      if (functionDeclarations.length > 0) {
        config.tools = [
          {
            functionDeclarations
          }
        ];
      }

      // CRITICAL: NEVER set responseSchema per ADR-0001
      // The schema-in-prompt contract lives in the prompt; validation enforces shape.

      // Resolve the client for this call
      const actualClient: GoogleGenAI = hasClient ? (options.client as GoogleGenAI) : client;

      try {
        // Race the SDK call against the timeout
        const response: GenerateContentResponse = await rejectAfterTimeout(
          actualClient.models.generateContent({
            model: req.model,
            contents,
            config
          }),
          effectiveTimeoutMs,
          `Gemini transport timeout after ${effectiveTimeoutMs}ms`
        );

        // Extract response data
        const candidate = response.candidates?.[0];
        const finishReason = candidate?.finishReason;
        const parts = candidate?.content?.parts;

        // Extract usage metadata when present
        const usage = response.usageMetadata;
        let usageMetadata: LlmResponse['usage'] | undefined;
        if (usage) {
          usageMetadata = {
            outputTokens: usage.candidatesTokenCount,
            thoughtsTokenCount: usage.thoughtsTokenCount
          };
        }

        // Extract tool call (if any)
        const functionCalls = candidate?.content?.parts?.filter(
          (part): part is Part & { functionCall: FunctionCall } =>
            'functionCall' in part && part.functionCall !== undefined
        ).map((part): FunctionCall => part.functionCall);

        const toolCall = extractFirstFunctionCall(functionCalls);

        // Extract raw text (join text parts if multiple)
        const rawText = joinTextParts(parts);

        // Build the response
        const llmResponse: LlmResponse = {
          rawText,
          finishReason,
          ...(toolCall !== undefined ? { toolCall } : {}),
          ...(usageMetadata !== undefined ? { usage: usageMetadata } : {})
        };

        return llmResponse;
      } catch (err) {
        // Rethrow SDK errors unchanged (the harness classifies them)
        dbg.error('[GeminiTransport] SDK error rethrown', err);
        throw err;
      }
    }
  };

  return { ok: true, transport };
}
