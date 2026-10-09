/**
 * Agent Harness Transport Types
 *
 * Defines the interfaces for the LLM transport layer, trace structures, and
 * harness execution types. This is the public surface for embedding tool-calling
 * LLMs into backends via the harness executor.
 */

import type { EnvelopeState, ValidationFailure, AgentTurn } from '../envelope/envelope';
import type { ToolContract, ToolHandler } from '../tool/toolContract';

// ============================================================================
// LLM Transport Contract
// ============================================================================

/**
 * Thinking level for models that support reasoning traces.
 *
 * Narrowed in 0.3.0: `'minimal'` was removed because Gemini 3.7/3.8 Flash
 * hard-reject it (HTTP 400) — only `'low' | 'medium' | 'high'` are valid on
 * the models this library targets. Note that several models reject ANY
 * thinkingLevel, so the field stays opt-in (the transport omits it unless the
 * host explicitly configures it).
 */
export type HarnessThinkingLevel = 'low' | 'medium' | 'high';

/**
 * Request sent to an LLM transport during a harness turn.
 *
 * All prompts and tool declarations are provided to the transport; the
 * harness composes the prompt text but never interprets model output — the
 * transport returns raw text and optional tool call data.
 */
export interface LlmRequest {
  /** Model identifier to invoke (chosen from the harness's fallback chain) */
  readonly model: string;
  /** System instruction (agent behavior prompt) */
  readonly systemInstruction: string;
  /** Full prompt text (state snapshot + contract + iteration context) */
  readonly promptText: string;
  /** Tool contracts declared to the model (the "belt") */
  readonly toolDeclarations: ReadonlyArray<ToolContract>;
  /**
   * Sampling temperature.
   *
   * @deprecated Never transmitted since 0.3.0: upcoming Gemini models error on
   * `temperature`/`top_p`/`top_k`, so the harness no longer forwards it and
   * the Gemini transport no longer maps it. Retained on the request type for
   * source compatibility only; also excluded from the record/replay hash.
   */
  readonly temperature?: number;
  /** Maximum output tokens (default 8192) */
  readonly maxOutputTokens?: number;
  /** Thinking level for models that support reasoning traces */
  readonly thinkingLevel?: HarnessThinkingLevel;
  /** Inline data for vision inputs (image/png, image/jpeg, etc.) */
  readonly inlineData?: {
    readonly mimeType: string;
    readonly data: string;
  };
}

/**
 * Raw response from an LLM transport.
 *
 * The transport returns the raw model text and any tool call it requested.
 * The harness extracts JSON (when no tool call) or validates tool args before
 * invoking handlers.
 */
export interface LlmResponse {
  /** Raw text output from the model (may be fenced JSON, prose, or mixed) */
  readonly rawText: string;
  /** Tool call requested by the model (present iff the model made a tool call) */
  readonly toolCall?: { name: string; args: unknown };
  /** Provider finish reason ('STOP', 'MAX_TOKENS', 'RECITATION', etc.) */
  readonly finishReason?: string;
  /** Token usage metadata (for telemetry and budget tracking) */
  readonly usage?: {
    readonly outputTokens?: number;
    readonly thoughtsTokenCount?: number;
  };
}

/**
 * Abstract LLM transport interface — the harness's model access layer.
 *
 * Implementations wrap specific providers (Gemini, OpenAI, Claude, etc.) and
 * return a unified {@link LlmResponse}. May reject; the harness catches and
 * classifies errors (transient vs permanent).
 */
export interface LlmTransport {
  /**
   * Invokes an LLM with the given request and timeout.
   *
   * @param req - The request to send to the model
   * @param opts - Execution options including timeout
   * @returns Resolved model response or rejects on error
   * @throws {Error} Network errors, provider failures (harness catches)
   */
  complete(req: LlmRequest, opts: { timeoutMs: number }): Promise<LlmResponse>;
}

// ============================================================================
// Tool Binding
// ============================================================================

/**
 * Binds a tool contract to its handler, ready for harness registration.
 *
 * The harness validates arguments against `contract.inputSchema` BEFORE
 * calling `handler` (MCP duty: validate arguments before execution).
 *
 * @template I - Input type (validated against `contract.inputSchema`)
 * @template R - Structured output type (validated against `contract.outputSchema` when declared)
 */
export interface ToolBinding<I = unknown, R = unknown> {
  /** Declarative tool contract (name, schemas, annotations) */
  readonly contract: ToolContract<I, R>;
  /** Handler invoked after argument validation passes */
  readonly handler: ToolHandler<I, R>;
}

// ============================================================================
// Trace & Telemetry
// ============================================================================

/**
 * Context provided when all fallback models are exhausted.
 *
 * The host receives this context and returns a safe {@link AgentTurn<never>}
 * (typically a question envelope asking the user for clarification).
 */
export interface FailureContext {
  /** Correlation ID for this turn */
  readonly correlationId: string;
  /** Last validation failure that triggered fallback */
  readonly lastFailure: ValidationFailure;
  /** Full trace leading to exhaustion */
  readonly trace: TurnTrace;
}

/**
 * Discriminated union of trace entry types.
 *
 * Each entry represents a discrete event in a turn: model calls, tool
 * executions, repair attempts, fallback decisions, etc.
 */
export type TraceEntry =
  | { readonly kind: 'model_call'; readonly model: string; readonly attempt: number; readonly latencyMs: number; readonly rawCharCount: number; readonly snippet: string }
  | { readonly kind: 'model_error'; readonly model: string; readonly attempt: number; readonly error: string; readonly status?: number | string; readonly isTransient: boolean }
  | { readonly kind: 'tool_call'; readonly name: string; readonly ok: boolean; readonly attempt: number }
  | {
      readonly kind: 'repair';
      readonly attempt: number;
      readonly outcome: 'attempted' | 'succeeded' | 'failed' | 'skipped_budget' | 'failed_extraction' | 'failed_validation';
      readonly detail?: string;
    }
  | { readonly kind: 'fallback'; readonly fromModel: string; readonly toModel: string }
  | {
      readonly kind: 'decision';
      readonly decision:
        | 'tool_call_wins_precedence'
        | 'repeated_tool_call_flagged'
        | 'tool_skipped_final_iteration'
        | 'degeneration_abort'
        | 'repair_skipped_degenerate'
        | 'degeneration_config_error'
        | 'transient_backoff'
        | 'extraction_failed'
        | 'validation_failed'
        | 'budget_exhausted'
        | 'harness_threw';
      readonly detail?: string;
    };

/**
 * Full execution trace for a harness turn.
 *
 * Records all model calls, tool executions, decisions, and timing. The trace
 * is returned in both success and failure results for observability.
 */
export interface TurnTrace {
  /** Correlation ID for this turn (all entries share this ID) */
  readonly correlationId: string;
  /** Chronological trace entries */
  readonly entries: ReadonlyArray<TraceEntry>;
  /** Turn start timestamp (ms since epoch, from injected `now()`) */
  readonly startedAt: number;
  /** Turn end timestamp (ms since epoch, from injected `now()`) */
  readonly endedAt: number;
}

/**
 * Telemetry events emitted by the harness during execution.
 *
 * Every event carries a correlation ID and typed payload; hosts install a
 * listener via {@link HarnessOptions.onTelemetry} for observability.
 */
export type HarnessTelemetryEvent =
  | { readonly type: 'model_call'; readonly correlationId: string; readonly model: string; readonly attempt: number; readonly latencyMs: number; readonly rawCharCount: number; readonly snippet: string }
  | { readonly type: 'model_error'; readonly correlationId: string; readonly model: string; readonly attempt: number; readonly error: string; readonly status?: number | string; readonly isTransient: boolean }
  | { readonly type: 'validated'; readonly correlationId: string; readonly state: EnvelopeState; readonly attempt: number }
  | { readonly type: 'repair_attempted' | 'repair_succeeded'; readonly correlationId: string; readonly attempt: number }
  | { readonly type: 'degenerate_output'; readonly correlationId: string; readonly model: string; readonly signatures: readonly string[] }
  | { readonly type: 'ambiguous_response'; readonly correlationId: string; readonly model: string }
  | { readonly type: 'transient_error'; readonly correlationId: string; readonly model: string; readonly backoffMs: number }
  | { readonly type: 'fallback_model'; readonly correlationId: string; readonly fromModel: string; readonly toModel: string }
  | { readonly type: 'budget_exhausted' | 'iterations_exhausted' | 'tool_calls_exhausted' | 'all_fallbacks_exhausted'; readonly correlationId: string };

// ============================================================================
// Iteration Context
// ============================================================================

/**
 * Dynamic iteration context injected into prompts.
 *
 * The harness passes this to {@link HarnessOptions.onIterationContext} (if
 * provided) to render a dynamic budget/status line for each model call.
 */
export interface IterationContext {
  /** Current iteration number (1-based) */
  readonly iteration: number;
  /** Maximum iterations permitted for this turn */
  readonly maxIterations: number;
  /** Remaining iterations (including this one) */
  readonly remainingIterations: number;
  /** Remaining tool calls total across all iterations */
  readonly remainingToolCalls: number;
  /** True if this is the final permitted iteration */
  readonly isFinalIteration: boolean;
}

// ============================================================================
// Harness Input & Result Types
// ============================================================================

/**
 * Input provided to the harness for a single turn.
 *
 * The harness composes the full prompt from these parts: system instruction
 * (host) + contract prompt (derived from manifest) + host prompt text + state
 * snapshot + iteration budget line + conversation context (if any).
 */
export interface HarnessTurnInput {
  /** System instruction for the model (agent behavior) */
  readonly systemInstruction: string;
  /** Host prompt text (current state, user command, etc.) */
  readonly promptText: string;
  /** Optional conversation history (prior turns for context) */
  readonly conversationContext?: string;
}

/**
 * Typed failure kinds for a harness turn.
 *
 * - `exhausted` / `budget` / `all_fallbacks` — policy outcomes (no valid
 *   envelope obtained within iteration, time, or model-chain limits).
 * - `config` — hard harness-configuration failure (e.g. invalid
 *   degeneration-detection options); must not be retried or treated as model
 *   behavior.
 * - `harness_threw` — the executor itself threw (an injected dependency —
 *   `sleep`, `onFallbackExhausted`, `onIterationContext`, `now`, `idFactory` —
 *   rejected or threw). The wrapper converts this to a typed failure so
 *   `runTurn` NEVER rejects; hosts switching on `kind` exhaustively are
 *   compile-forced to handle it.
 */
export type HarnessFailureKind = 'exhausted' | 'budget' | 'all_fallbacks' | 'config' | 'harness_threw';

/**
 * Result of a harness turn execution.
 *
 * Success (`ok: true`) returns a validated {@link AgentTurn<P>}` with trace.
 * Failure (`ok: false`) returns a typed {@link HarnessFailureKind} with an
 * {@link AgentTurn<never>} (question envelope) and trace.
 *
 * @template P - Payload type (only present when state === 'proposal')
 */
export type HarnessResult<P> =
  | { readonly ok: true; readonly turn: AgentTurn<P>; readonly trace: TurnTrace }
  | { readonly ok: false; readonly kind: HarnessFailureKind; readonly turn: AgentTurn<never>; readonly trace: TurnTrace };

/**
 * Agent harness interface — the public entry point for tool-calling LLM turns.
 *
 * The harness manages model fallbacks, iteration budgets, tool execution,
 * repair attempts, degeneration detection, transient backoff, and telemetry.
 * It never rejects — all errors, including throws from injected dependencies,
 * become typed failure results.
 *
 * @template P - Payload type (only present when state === 'proposal')
 */
export interface AgentHarness<P> {
  /**
   * Executes a single turn with the given input.
   *
   * @param input - Turn input (system instruction, prompt text, optional conversation context)
   * @returns Validated turn on success, typed failure on exhaustion/budget/fallback/harness-throw — never rejects
   */
  runTurn(input: HarnessTurnInput): Promise<HarnessResult<P>>;
}
