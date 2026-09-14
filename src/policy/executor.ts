/**
 * Agent Harness Executor — Policy Loop Implementation
 *
 * Provides the core harness executor that manages model fallbacks, iteration
 * budgets, tool execution, repair attempts, degeneration detection, transient
 * backoff, and telemetry. This is the policy engine that drives tool-calling
 * LLM turns with deterministic behavior and record/replay support.
 *
 * Ported from courtpuzzle/functions/src/providers/copilotProvider.ts.
 */

import type { Contract } from '../contract/contract';
import type { ToolResult } from '../tool/toolContract';
import { validateToolArgs } from '../tool/toolContract';
import { validateEnvelopeOutput } from '../envelope/validate';
import { extractJson, diagnosticSnippet } from '../json/extract';
import { detectDegeneration } from '../diagnostics/degeneration';
import { isTransientProviderError } from './transientErrors';
import { dbg } from '../debug/debug';
import type {
  LlmTransport,
  LlmRequest,
  LlmResponse,
  ToolBinding,
  FailureContext,
  TraceEntry,
  HarnessTelemetryEvent,
  IterationContext,
  HarnessTurnInput,
  HarnessResult,
  AgentHarness
} from '../transport/types';
import type { AgentTurn } from '../envelope/envelope';

// ============================================================================
// Default Constants
// ============================================================================

/**
 * Default maximum number of iterations per turn.
 * `maxIterations: 1` == single model call, no tool feedback.
 */
export const DEFAULT_MAX_ITERATIONS = 1;

/**
 * Default timeout for a single model call.
 */
export const DEFAULT_PER_MODEL_TIMEOUT_MS = 15_000;

/**
 * Default total time budget for a complete turn.
 */
export const DEFAULT_TOTAL_BUDGET_MS = 45_000;

/**
 * Default maximum number of repair attempts per turn.
 */
export const DEFAULT_MAX_REPAIRS = 1;

/**
 * Default number of degeneration detections before aborting the fallback chain.
 */
export const DEFAULT_MAX_DEGENERATIONS_BEFORE_ABORT = 2;

/**
 * Default backoff delay for transient errors.
 */
export const DEFAULT_TRANSIENT_BACKOFF_MS = 1_000;

/**
 * Default maximum retries for transient errors per model.
 */
export const DEFAULT_MAX_TRANSIENT_RETRIES_PER_MODEL = 3;

/**
 * Default maximum tool calls per iteration.
 */
export const DEFAULT_MAX_TOOL_CALLS_PER_ITERATION = 10;

/**
 * Default total tool calls across all iterations.
 */
export const DEFAULT_MAX_TOTAL_TOOL_CALLS = 50;

/**
 * Default sampling temperature for model calls.
 */
export const DEFAULT_TEMPERATURE = 0.1;

/**
 * Default maximum output tokens for model calls.
 */
export const DEFAULT_MAX_OUTPUT_TOKENS = 8_192;

/**
 * Default thinking level for models that support reasoning traces.
 */
export const DEFAULT_THINKING_LEVEL: 'low' | 'minimal' = 'low';

/**
 * Default minimum remaining budget (ms) required to attempt a repair call.
 * Below this, the harness skips repair and falls back instead of starting a
 * model call that would be cut off mid-flight.
 */
export const DEFAULT_MIN_BUDGET_FOR_REPAIR_MS = 5_000;

// ============================================================================
// Harness Options
// ============================================================================

/**
 * Configuration options for {@link createHarness}.
 *
 * All timeouts and budgets are positive finite numbers; iteration limits are
 * >= 1. Invalid configuration produces a typed failure, never a throw.
 */
export interface HarnessOptions {
  /** LLM transport implementation (wraps a specific provider) */
  readonly transport: LlmTransport;
  /** Fallback model chain (attempted in order until one succeeds) */
  readonly models: string[];
  /** Maximum iterations per turn (default DEFAULT_MAX_ITERATIONS, >= 1) */
  readonly maxIterations?: number;
  /** Maximum tool calls per iteration (default DEFAULT_MAX_TOOL_CALLS_PER_ITERATION) */
  readonly maxToolCallsPerIteration?: number;
  /** Maximum total tool calls across all iterations (default DEFAULT_MAX_TOTAL_TOOL_CALLS) */
  readonly maxTotalToolCalls?: number;
  /** Optional tool bindings (handlers for tool calls) */
  readonly tools?: ReadonlyArray<ToolBinding>;
  /** Timeout per model call (default DEFAULT_PER_MODEL_TIMEOUT_MS) */
  readonly perModelTimeoutMs?: number;
  /** Total time budget for the turn (default DEFAULT_TOTAL_BUDGET_MS) */
  readonly totalBudgetMs?: number;
  /** Sampling temperature (default DEFAULT_TEMPERATURE, 0.0..2.0) */
  readonly temperature?: number;
  /** Maximum output tokens (default DEFAULT_MAX_OUTPUT_TOKENS, positive integer) */
  readonly maxOutputTokens?: number;
  /** Thinking level for models with reasoning traces (default DEFAULT_THINKING_LEVEL) */
  readonly thinkingLevel?: 'low' | 'minimal';
  /** Maximum repair attempts (default DEFAULT_MAX_REPAIRS) */
  readonly maxRepairs?: number;
  /** Degenerations before abort (default DEFAULT_MAX_DEGENERATIONS_BEFORE_ABORT) */
  readonly maxDegenerationsBeforeAbort?: number;
  /** Backoff for transient errors (default DEFAULT_TRANSIENT_BACKOFF_MS) */
  readonly transientBackoffMs?: number;
  /** Minimum remaining budget to attempt a repair call (default DEFAULT_MIN_BUDGET_FOR_REPAIR_MS) */
  readonly minBudgetForRepairMs?: number;
  /** Maximum retries for transient errors per model (default DEFAULT_MAX_TRANSIENT_RETRIES_PER_MODEL) */
  readonly maxTransientRetriesPerModel?: number;
  /** Injected clock (tests use fake time, production uses Date.now) */
  readonly now?: () => number;
  /** Injected ID factory (tests use deterministic IDs) */
  readonly idFactory?: () => string;
  /** Telemetry event listener (optional, for observability) */
  readonly onTelemetry?: (e: HarnessTelemetryEvent) => void;
  /** Dynamic iteration context renderer (optional, for budget prompts) */
  readonly onIterationContext?: (ctx: IterationContext) => string;
  /** Host fallback handler (optional, called when all models exhausted) */
  readonly onFallbackExhausted?: (ctx: FailureContext) => AgentTurn<never>;
  /** Injected sleep function (tests use spies, production uses setTimeout) */
  readonly sleep?: (ms: number) => Promise<void>;
}

// ============================================================================
// Config Validation Types
// ============================================================================

/**
 * Machine-readable config validation failure codes.
 */
export type ConfigFailureCode =
  | 'models_empty'
  | 'max_iterations_invalid'
  | 'timeout_invalid'
  | 'budget_invalid'
  | 'temperature_invalid'
  | 'output_tokens_invalid'
  | 'thinking_level_invalid'
  | 'repairs_invalid'
  | 'degenerations_invalid'
  | 'backoff_invalid'
  | 'repair_budget_invalid'
  | 'retries_invalid'
  | 'tool_caps_invalid';

/**
 * One config validation failure.
 */
export interface ConfigFailure {
  /** Machine-readable failure code */
  readonly code: ConfigFailureCode;
  /** Human-readable failure message */
  readonly message: string;
}

/**
 * Result of {@link createHarness}: success or typed failures — never a throw.
 *
 * @template P - Payload type (only present when state === 'proposal')
 */
export type CreateHarnessResult<P> =
  | { readonly ok: true; readonly harness: AgentHarness<P> }
  | { readonly ok: false; readonly failures: readonly ConfigFailure[] };

// ============================================================================
// Internal State Types
// ============================================================================

/**
 * Internal harness state for a single turn.
 */
interface HarnessState {
  /** Correlation ID for this turn */
  readonly correlationId: string;
  /** Turn start timestamp */
  readonly startedAt: number;
  /** Trace entries accumulator */
  readonly traceEntries: TraceEntry[];
  /** Remaining iterations */
  remainingIterations: number;
  /** Remaining total tool calls */
  remainingToolCalls: number;
  /** Remaining repairs */
  remainingRepairs: number;
  /** Tool calls seen this iteration (for repeated-call detection) */
  toolCallsThisIteration: Set<string>;
  /** Degeneration count per model */
  degenerationCounts: Map<string, number>;
  /** Transient retry counts per model */
  transientRetryCounts: Map<string, number>;
  /** Tool results accumulated so far (for next iteration's prompt) */
  toolResults: Array<{ name: string; args: unknown; result: ToolResult }>;
}

/**
 * Normalized candidate output (tool call or extracted JSON).
 */
interface NormalizedCandidate {
  /** Tool call if present */
  readonly toolCall?: { name: string; args: unknown };
  /** Extracted JSON if no tool call */
  readonly extractedJson?: unknown;
  /** Whether this was an ambiguous response (both tool call and JSON) */
  readonly ambiguous: boolean;
}

/**
 * Extracts a human-readable message from an unknown error value (Error instance,
 * ReplayMissError object, structured API error, or primitive).
 */
function formatErrorMessage(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  if (typeof err === 'object' && err !== null) {
    if ('message' in err && typeof (err as { message: unknown }).message === 'string') {
      return (err as { message: string }).message;
    }
    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }
  return String(err);
}

// ============================================================================
// Config Validation
// ============================================================================

/**
 * Validates harness options and returns failures or `ok: true`.
 *
 * @param options - Candidate options (untrusted boundary input)
 * @returns All failures found (empty = valid)
 */
function validateOptions(options: HarnessOptions): readonly ConfigFailure[] {
  const failures: ConfigFailure[] = [];

  // Models must be a non-empty array
  if (!Array.isArray(options.models) || options.models.length === 0) {
    failures.push({ code: 'models_empty', message: 'Model list must be a non-empty array.' });
  }

  // Max iterations must be >= 1
  const maxIter = options.maxIterations;
  if (maxIter !== undefined && (typeof maxIter !== 'number' || !Number.isFinite(maxIter) || maxIter < 1)) {
    failures.push({ code: 'max_iterations_invalid', message: 'maxIterations must be a number >= 1.' });
  }

  // Timeout must be positive finite
  const timeout = options.perModelTimeoutMs;
  if (timeout !== undefined && (typeof timeout !== 'number' || !Number.isFinite(timeout) || timeout <= 0)) {
    failures.push({ code: 'timeout_invalid', message: 'perModelTimeoutMs must be a positive finite number.' });
  }

  // Budget must be positive finite
  const budget = options.totalBudgetMs;
  if (budget !== undefined && (typeof budget !== 'number' || !Number.isFinite(budget) || budget <= 0)) {
    failures.push({ code: 'budget_invalid', message: 'totalBudgetMs must be a positive finite number.' });
  }

  // Repairs must be >= 0
  const repairs = options.maxRepairs;
  if (repairs !== undefined && (typeof repairs !== 'number' || !Number.isInteger(repairs) || repairs < 0)) {
    failures.push({ code: 'repairs_invalid', message: 'maxRepairs must be a non-negative integer.' });
  }

  // Degenerations must be >= 1
  const degenerations = options.maxDegenerationsBeforeAbort;
  if (degenerations !== undefined && (typeof degenerations !== 'number' || !Number.isInteger(degenerations) || degenerations < 1)) {
    failures.push({ code: 'degenerations_invalid', message: 'maxDegenerationsBeforeAbort must be a positive integer.' });
  }

  // Backoff must be non-negative
  const backoff = options.transientBackoffMs;
  if (backoff !== undefined && (typeof backoff !== 'number' || !Number.isFinite(backoff) || backoff < 0)) {
    failures.push({ code: 'backoff_invalid', message: 'transientBackoffMs must be a non-negative number.' });
  }

  // Repair budget floor must be non-negative
  const repairFloor = options.minBudgetForRepairMs;
  if (repairFloor !== undefined && (typeof repairFloor !== 'number' || !Number.isFinite(repairFloor) || repairFloor < 0)) {
    failures.push({ code: 'repair_budget_invalid', message: 'minBudgetForRepairMs must be a non-negative number.' });
  }

  // Transient retries must be >= 1
  const retries = options.maxTransientRetriesPerModel;
  if (retries !== undefined && (typeof retries !== 'number' || !Number.isInteger(retries) || retries < 1)) {
    failures.push({ code: 'retries_invalid', message: 'maxTransientRetriesPerModel must be a positive integer.' });
  }

  // Tool caps must be non-negative
  const maxPerIter = options.maxToolCallsPerIteration;
  if (maxPerIter !== undefined && (typeof maxPerIter !== 'number' || !Number.isInteger(maxPerIter) || maxPerIter < 0)) {
    failures.push({ code: 'tool_caps_invalid', message: 'maxToolCallsPerIteration must be a non-negative integer.' });
  }

  const maxTotal = options.maxTotalToolCalls;
  if (maxTotal !== undefined && (typeof maxTotal !== 'number' || !Number.isInteger(maxTotal) || maxTotal < 0)) {
    failures.push({ code: 'tool_caps_invalid', message: 'maxTotalToolCalls must be a non-negative integer.' });
  }

  // Temperature must be finite and in range [0, 2]
  const temp = options.temperature;
  if (temp !== undefined && (typeof temp !== 'number' || !Number.isFinite(temp) || temp < 0 || temp > 2)) {
    failures.push({ code: 'temperature_invalid', message: 'temperature must be a finite number in range [0, 2].' });
  }

  // Max output tokens must be positive integer
  const maxTokens = options.maxOutputTokens;
  if (maxTokens !== undefined && (typeof maxTokens !== 'number' || !Number.isInteger(maxTokens) || maxTokens <= 0)) {
    failures.push({ code: 'output_tokens_invalid', message: 'maxOutputTokens must be a positive integer.' });
  }

  // Thinking level must be valid
  const thinking = options.thinkingLevel;
  if (thinking !== undefined && thinking !== 'low' && thinking !== 'minimal') {
    failures.push({ code: 'thinking_level_invalid', message: 'thinkingLevel must be "low" or "minimal".' });
  }

  return failures;
}

// ============================================================================
// Harness Implementation
// ============================================================================

/**
 * Creates an agent harness with the given contract and options.
 *
 * On config validation failure, returns a typed result — never throws.
 *
 * @template P - Payload type (only present when state === 'proposal')
 * @param options - Harness configuration (transport, models, budgets, etc.)
 * @param contract - Validated contract for payload validation
 * @returns `{ ok: true, harness }` or `{ ok: false, failures }`
 */
export function createHarness<P>(
  options: HarnessOptions,
  contract: Contract<P>
): CreateHarnessResult<P> {
  const failures = validateOptions(options);
  if (failures.length > 0) {
    return { ok: false, failures };
  }

  const resolved: Required<
    Omit<
      HarnessOptions,
      'tools' | 'onTelemetry' | 'onIterationContext' | 'onFallbackExhausted' | 'thinkingLevel'
    >
  > & { thinkingLevel?: 'low' | 'minimal' } = {
    transport: options.transport,
    models: options.models,
    maxIterations: options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    maxToolCallsPerIteration: options.maxToolCallsPerIteration ?? DEFAULT_MAX_TOOL_CALLS_PER_ITERATION,
    maxTotalToolCalls: options.maxTotalToolCalls ?? DEFAULT_MAX_TOTAL_TOOL_CALLS,
    perModelTimeoutMs: options.perModelTimeoutMs ?? DEFAULT_PER_MODEL_TIMEOUT_MS,
    totalBudgetMs: options.totalBudgetMs ?? DEFAULT_TOTAL_BUDGET_MS,
    maxRepairs: options.maxRepairs ?? DEFAULT_MAX_REPAIRS,
    maxDegenerationsBeforeAbort: options.maxDegenerationsBeforeAbort ?? DEFAULT_MAX_DEGENERATIONS_BEFORE_ABORT,
    transientBackoffMs: options.transientBackoffMs ?? DEFAULT_TRANSIENT_BACKOFF_MS,
    minBudgetForRepairMs: options.minBudgetForRepairMs ?? DEFAULT_MIN_BUDGET_FOR_REPAIR_MS,
    maxTransientRetriesPerModel: options.maxTransientRetriesPerModel ?? DEFAULT_MAX_TRANSIENT_RETRIES_PER_MODEL,
    temperature: options.temperature ?? DEFAULT_TEMPERATURE,
    maxOutputTokens: options.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
    // Opt-in: the Gemini API rejects thinkingLevel for several models
    // (HTTP 400 "Thinking level is not supported for this model"), so the
    // transport must omit it unless the host explicitly configures it.
    thinkingLevel: options.thinkingLevel,
    now: options.now ?? (() => Date.now()),
    idFactory: options.idFactory ?? (() => `harness_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`),
    sleep: options.sleep ?? ((ms: number) => new Promise(resolve => setTimeout(resolve, ms)))
  };

  const harness: AgentHarness<P> = {
    async runTurn(input: HarnessTurnInput): Promise<HarnessResult<P>> {
      const correlationId = resolved.idFactory();
      const startedAt = resolved.now();

      const state: HarnessState = {
        correlationId,
        startedAt,
        traceEntries: [],
        remainingIterations: resolved.maxIterations,
        remainingToolCalls: resolved.maxTotalToolCalls,
        remainingRepairs: resolved.maxRepairs,
        toolCallsThisIteration: new Set(),
        degenerationCounts: new Map(),
        transientRetryCounts: new Map(),
        toolResults: []
      };

      // Emit telemetry safely (listener throws never break the turn)
      const emit = (e: HarnessTelemetryEvent): void => {
        try {
          options.onTelemetry?.(e);
        } catch (err) {
          dbg.warn('Telemetry listener threw:', err);
        }
      };

      // Add trace entry
      const addTrace = (entry: TraceEntry): void => {
        state.traceEntries.push(entry);
      };

      // Build prompt with tool results from previous iteration
      const buildPrompt = (): string => {
        let prompt = input.promptText;

        // Append tool results from previous iteration
        if (state.toolResults.length > 0) {
          const toolResultsText = state.toolResults
            .map(tr => {
              const isError = tr.result.isError ?? false;
              const content = tr.result.structuredContent ?? tr.result.text ?? '';
              // Include the call arguments: model calls are stateless, so
              // without them the model cannot correlate results with calls.
              return `Tool: ${tr.name}\nArguments: ${JSON.stringify(tr.args ?? null)}\nStatus: ${isError ? 'ERROR' : 'OK'}\nResult: ${JSON.stringify(content)}`;
            })
            .join('\n\n');
          prompt = `${prompt}\n\nPREVIOUS TOOL RESULTS:\n${toolResultsText}`;
        }

        return prompt;
      };

      // Compose full prompt with iteration budget
      const composeFullPrompt = (): string => {
        const prompt = buildPrompt();
        const contractText = contract.renderPromptContract();
        const iteration = resolved.maxIterations - state.remainingIterations + 1;
        const ctx: IterationContext = {
          iteration,
          maxIterations: resolved.maxIterations,
          remainingIterations: state.remainingIterations,
          remainingToolCalls: state.remainingToolCalls,
          isFinalIteration: state.remainingIterations === 1
        };
        const contextLine = options.onIterationContext?.(ctx) ?? '';
        const finalNote = ctx.isFinalIteration
          ? '\n\nThis is the final iteration — return a final envelope without tool calls.'
          : '';
        return `${contractText}\n\n${prompt}${contextLine}${finalNote}`;
      };

      // Normalize candidate from model response
      const normalizeCandidate = (resp: LlmResponse): NormalizedCandidate => {
        if (resp.toolCall) {
          const extracted = extractJson(resp.rawText);
          const ambiguous = extracted.ok && extracted.parsed !== undefined;
          return { toolCall: resp.toolCall, extractedJson: extracted.ok ? extracted.parsed : undefined, ambiguous };
        }
        const extracted = extractJson(resp.rawText);
        return { extractedJson: extracted.ok ? extracted.parsed : undefined, ambiguous: false };
      };

      // Default fallback handler (context provided but unused in default impl)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const defaultFallbackHandler = (_ctx: FailureContext): AgentTurn<never> => {
        return {
          envelope: {
            state: 'question',
            questionText: "I couldn't safely process that request. Could you please clarify or rephrase what you'd like to do?",
            explanation: 'I was unable to validate a response from the model.'
          }
        };
      };

      // Tool declarations are fixed for the whole turn
      const toolDeclarations = options.tools?.map(tb => tb.contract) ?? [];

      /**
       * Runs ONE repair attempt for a failed candidate output: builds the
       * repair prompt (full contract prompt + error preamble + bounded
       * diagnostic snippet of the failed raw output), calls the model once,
       * and validates the repaired candidate. The caller checks and decrements
       * the repair budget; every non-success outcome maps to "fall back to the
       * next model".
       */
      const attemptRepair = async (
        model: string,
        iteration: number,
        failureDescription: string,
        failedOutputSnippet: string
      ): Promise<{ readonly ok: true; readonly turn: AgentTurn<P> } | { readonly ok: false }> => {
        const repairPrompt = `${composeFullPrompt()}

The previous response failed validation:
Validation Error: ${failureDescription}

Previous response (bounded excerpt): ${failedOutputSnippet}

Please fix the error and return ONLY a valid JSON object following the output contract.`;

        const repairReq: LlmRequest = {
          model,
          systemInstruction: input.systemInstruction,
          promptText: repairPrompt,
          toolDeclarations,
          temperature: resolved.temperature,
          maxOutputTokens: resolved.maxOutputTokens,
          thinkingLevel: resolved.thinkingLevel
        };

        const callStart = resolved.now();
        try {
          const remainingBudget = resolved.totalBudgetMs - (callStart - startedAt);
          if (remainingBudget < resolved.minBudgetForRepairMs) {
            addTrace({ kind: 'repair', attempt: iteration, outcome: 'skipped_budget', detail: 'Insufficient budget for repair' });
            return { ok: false };
          }

          const timeoutMs = Math.min(resolved.perModelTimeoutMs, Math.max(remainingBudget, 1));
          const repairResp = await Promise.race([
            resolved.transport.complete(repairReq, { timeoutMs }),
            new Promise<LlmResponse>((_, reject) => resolved.sleep(timeoutMs).then(() => reject(new Error('Repair call timed out'))))
          ]);

          const repairCandidate = normalizeCandidate(repairResp);
          if (!repairCandidate.extractedJson) {
            addTrace({ kind: 'repair', attempt: iteration, outcome: 'failed_extraction' });
            return { ok: false };
          }

          const repairValidation = validateEnvelopeOutput(repairCandidate.extractedJson, contract.validateHostPayload, contract.manifest);
          if (!repairValidation.ok) {
            addTrace({ kind: 'repair', attempt: iteration, outcome: 'failed_validation', detail: repairValidation.failure.code });
            return { ok: false };
          }

          emit({ type: 'repair_succeeded', correlationId, attempt: iteration });
          addTrace({ kind: 'repair', attempt: iteration, outcome: 'succeeded' });
          return { ok: true, turn: repairValidation.value };
        } catch (err) {
          addTrace({ kind: 'repair', attempt: iteration, outcome: 'failed', detail: formatErrorMessage(err) });
          return { ok: false };
        }
      };

      // Main policy loop: model fallback chain
      for (let modelIdx = 0; modelIdx < resolved.models.length; modelIdx++) {
        const model = resolved.models[modelIdx];

        // Check budget before each model call
        const elapsed = resolved.now() - startedAt;
        if (elapsed >= resolved.totalBudgetMs) {
          emit({ type: 'budget_exhausted', correlationId });
          addTrace({ kind: 'decision', decision: 'budget_exhausted' });
          const failureContext: FailureContext = {
            correlationId,
            lastFailure: { code: 'malformed_json', message: 'Total time budget exhausted.' },
            trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
          };
          const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
          return {
            ok: false,
            kind: 'budget',
            turn: fallbackTurn,
            trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
          };
        }

        // Retry this model until success, repair exhausted, or degeneration abort
        let modelDegenerations = state.degenerationCounts.get(model) ?? 0;
        state.remainingIterations = resolved.maxIterations;
        state.toolCallsThisIteration.clear();

        while (state.remainingIterations > 0) {
          // Check iteration budget
          if (state.remainingIterations <= 0) {
            emit({ type: 'iterations_exhausted', correlationId });
            addTrace({ kind: 'decision', decision: 'tool_skipped_final_iteration' });
            const failureContext: FailureContext = {
              correlationId,
              lastFailure: { code: 'malformed_json', message: 'Maximum iterations reached.' },
              trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
            };
            const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
            return {
              ok: false,
              kind: 'exhausted',
              turn: fallbackTurn,
              trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
            };
          }

          const iteration = resolved.maxIterations - state.remainingIterations + 1;
          const isFinalIteration = state.remainingIterations === 1;

          // Check budget before model call (mid-model re-check)
          const elapsedBeforeCall = resolved.now() - startedAt;
          if (elapsedBeforeCall >= resolved.totalBudgetMs) {
            emit({ type: 'budget_exhausted', correlationId });
            addTrace({ kind: 'decision', decision: 'budget_exhausted', detail: 'Budget exhausted before model call' });
            const failureContext: FailureContext = {
              correlationId,
              lastFailure: { code: 'malformed_json', message: 'Total time budget exhausted.' },
              trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
            };
            const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
            return {
              ok: false,
              kind: 'budget',
              turn: fallbackTurn,
              trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
            };
          }

          // Compose request
          const req: LlmRequest = {
            model,
            systemInstruction: input.systemInstruction,
            promptText: composeFullPrompt(),
            toolDeclarations,
            temperature: resolved.temperature,
            maxOutputTokens: resolved.maxOutputTokens,
            thinkingLevel: resolved.thinkingLevel
          };

          const callStart = resolved.now();
          let resp: LlmResponse;

          // Call model with timeout
          try {
            const remainingBudget = resolved.totalBudgetMs - (resolved.now() - startedAt);
            const timeoutMs = Math.min(resolved.perModelTimeoutMs, Math.max(remainingBudget, 1));
            resp = await Promise.race([
              resolved.transport.complete(req, { timeoutMs }),
              new Promise<LlmResponse>((_, reject) => resolved.sleep(timeoutMs).then(() => reject(new Error('Model call timed out'))))
            ]);
          } catch (err) {
            dbg.warn('Model call error:', err);
            const isTransient = isTransientProviderError(err);
            const errorMessage = formatErrorMessage(err);
            const status = (err as { status?: number | string })?.status ?? (err as { code?: number | string })?.code;

            emit({
              type: 'model_error',
              correlationId,
              model,
              attempt: iteration,
              error: errorMessage,
              status,
              isTransient
            });
            addTrace({
              kind: 'model_error',
              model,
              attempt: iteration,
              error: errorMessage,
              status,
              isTransient
            });

            if (isTransient) {
              const currentRetries = state.transientRetryCounts.get(model) ?? 0;
              if (currentRetries >= resolved.maxTransientRetriesPerModel) {
                // Exceeded retries for this model -> fall back to next model
                break;
              }
              state.transientRetryCounts.set(model, currentRetries + 1);
              emit({ type: 'transient_error', correlationId, model, backoffMs: resolved.transientBackoffMs });
              addTrace({ kind: 'decision', decision: 'transient_backoff', detail: `Backing off ${resolved.transientBackoffMs}ms (retry ${currentRetries + 1}/${resolved.maxTransientRetriesPerModel})` });
              await resolved.sleep(resolved.transientBackoffMs);
              continue; // Retry same model
            }

            // Non-transient error -> fall back to next model
            break;
          }

          const latencyMs = resolved.now() - callStart;
          // Bug #3 fix: Redact snippet to prevent leaking user content in telemetry
          const snippet = diagnosticSnippet(resp.rawText, { redactSnippet: true });
          emit({
            type: 'model_call',
            correlationId,
            model,
            attempt: iteration,
            latencyMs,
            rawCharCount: resp.rawText.length,
            snippet
          });
          addTrace({
            kind: 'model_call',
            model,
            attempt: iteration,
            latencyMs,
            rawCharCount: resp.rawText.length,
            snippet
          });

          // Normalize candidate
          const candidate = normalizeCandidate(resp);

          // Emit ambiguous response telemetry
          if (candidate.ambiguous) {
            emit({ type: 'ambiguous_response', correlationId, model });
            addTrace({ kind: 'decision', decision: 'tool_call_wins_precedence', detail: 'Both tool call and JSON present; tool call wins' });
          }

          // Check degeneration (on raw text, before validation)
          const degenerationResult = detectDegeneration({
            rawText: resp.rawText,
            finishReason: resp.finishReason
          });

          // A config failure here is a harness configuration bug — a hard
          // typed failure. It is NOT degeneration: no degeneration counter is
          // consumed, no retry, no model fallback.
          if (!degenerationResult.ok) {
            dbg.error('Degeneration detection config failure:', degenerationResult.failure.message);
            addTrace({
              kind: 'decision',
              decision: 'degeneration_config_error',
              detail: degenerationResult.failure.message
            });
            const failureContext: FailureContext = {
              correlationId,
              lastFailure: { code: 'schema_invalid', message: `Harness configuration error: ${degenerationResult.failure.message}` },
              trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
            };
            const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
            return {
              ok: false,
              kind: 'config',
              turn: fallbackTurn,
              trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
            };
          }

          const degeneration = degenerationResult.verdict;

          if (degeneration.degenerate) {
            emit({
              type: 'degenerate_output',
              correlationId,
              model,
              signatures: degeneration.signatures
            });
            modelDegenerations++;
            state.degenerationCounts.set(model, modelDegenerations);

            if (modelDegenerations >= resolved.maxDegenerationsBeforeAbort) {
              addTrace({ kind: 'decision', decision: 'degeneration_abort', detail: `Model ${model} degenerated ${modelDegenerations} times` });
              break; // Fall back to next model
            }

            // Degenerate output skips repair -> try next iteration (same model)
            addTrace({ kind: 'decision', decision: 'repair_skipped_degenerate' });
            state.remainingIterations--;
            continue;
          }

          // Handle tool call
          if (candidate.toolCall) {
            // Check if on final iteration
            if (isFinalIteration) {
              addTrace({ kind: 'decision', decision: 'tool_skipped_final_iteration' });
              const failureContext: FailureContext = {
                correlationId,
                lastFailure: { code: 'malformed_json', message: 'Tool requested on final iteration.' },
                trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
              };
              const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
              return {
                ok: false,
                kind: 'exhausted',
                turn: fallbackTurn,
                trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
              };
            }

            // Check tool caps
            if (state.remainingToolCalls <= 0) {
              emit({ type: 'tool_calls_exhausted', correlationId });
              addTrace({ kind: 'decision', decision: 'tool_skipped_final_iteration', detail: 'Tool call budget exhausted, skipping tool execution' });
              // Skip tool execution but continue processing model responses
              state.remainingIterations--;
              continue;
            }

            const toolName = candidate.toolCall.name;
            const toolArgs = candidate.toolCall.args;

            // Detect repeated tool call
            const callKey = `${toolName}:${JSON.stringify(toolArgs)}`;
            if (state.toolCallsThisIteration.has(callKey)) {
              addTrace({ kind: 'decision', decision: 'repeated_tool_call_flagged', detail: `Tool ${toolName} called again with same args` });
            }
            state.toolCallsThisIteration.add(callKey);

            // Find tool binding
            const toolBinding = options.tools?.find(tb => tb.contract.name === toolName);
            if (!toolBinding) {
              // Unknown tool -> synthesize error result
              state.toolResults.push({
                name: toolName,
                args: toolArgs,
                result: { isError: true, text: `Unknown tool: ${toolName}` }
              });
              state.remainingToolCalls--;
              state.remainingIterations--;
              addTrace({ kind: 'tool_call', name: toolName, ok: false, attempt: iteration });
              continue; // Next iteration with this result
            }

            // Validate tool args
            const argsValidation = validateToolArgs(toolBinding.contract, toolArgs);
            if (!argsValidation.ok) {
              // Args invalid -> synthesize error result, don't invoke handler
              const violationsText = argsValidation.violations.map(v => `[${v.path}] ${v.message}`).join(', ');
              state.toolResults.push({
                name: toolName,
                args: toolArgs,
                result: { isError: true, text: `Invalid arguments: ${violationsText}` }
              });
              state.remainingToolCalls--;
              state.remainingIterations--;
              addTrace({ kind: 'tool_call', name: toolName, ok: false, attempt: iteration });
              continue; // Next iteration with this result
            }

            // Check budget before tool handler (mid-handler re-check)
            const elapsedBeforeTool = resolved.now() - startedAt;
            if (elapsedBeforeTool >= resolved.totalBudgetMs) {
              emit({ type: 'budget_exhausted', correlationId });
              addTrace({ kind: 'decision', decision: 'budget_exhausted', detail: 'Budget exhausted before tool handler' });
              const failureContext: FailureContext = {
                correlationId,
                lastFailure: { code: 'malformed_json', message: 'Total time budget exhausted.' },
                trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
              };
              const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
              return {
                ok: false,
                kind: 'budget',
                turn: fallbackTurn,
                trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
              };
            }

            // Invoke tool handler
            try {
              const toolResult = await Promise.race([
                toolBinding.handler(toolArgs),
                new Promise<ToolResult>((_, reject) => resolved.sleep(resolved.perModelTimeoutMs).then(() => reject(new Error('Tool handler timed out'))))
              ]);

              state.toolResults.push({ name: toolName, args: toolArgs, result: toolResult });
              state.remainingToolCalls--;
              state.remainingIterations--;
              addTrace({ kind: 'tool_call', name: toolName, ok: !(toolResult.isError ?? false), attempt: iteration });

              // Continue to next iteration with tool result
              continue;
            } catch (err) {
              // Handler threw -> in-band error result
              const errorText = formatErrorMessage(err);
              state.toolResults.push({
                name: toolName,
                args: toolArgs,
                result: { isError: true, text: `Tool execution failed: ${errorText}` }
              });
              state.remainingToolCalls--;
              state.remainingIterations--;
              addTrace({ kind: 'tool_call', name: toolName, ok: false, attempt: iteration });
              continue; // Next iteration with this result
            }
          }

          // No tool call -> validate envelope
          const payload = candidate.extractedJson;
          if (!payload) {
            addTrace({ kind: 'decision', decision: 'extraction_failed', detail: `Failed to extract JSON from raw response (length: ${resp.rawText.length})` });
            // No valid candidate -> repair
            if (state.remainingRepairs > 0) {
              state.remainingRepairs--;
              emit({ type: 'repair_attempted', correlationId, attempt: iteration });
              addTrace({ kind: 'repair', attempt: iteration, outcome: 'attempted' });

              const repair = await attemptRepair(
                model,
                iteration,
                'No valid JSON object could be extracted from the response. Please provide a valid JSON response.',
                snippet
              );
              if (repair.ok) {
                return {
                  ok: true,
                  turn: repair.turn,
                  trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
                };
              }
            }

            // Repair impossible or failed -> fall back to next model
            break;
          }

          // Validate envelope
          const validation = validateEnvelopeOutput(payload, contract.validateHostPayload, contract.manifest);
          if (!validation.ok) {
            addTrace({ kind: 'decision', decision: 'validation_failed', detail: `${validation.failure.code}: ${validation.failure.message}` });
            // Validation failed -> repair
            if (state.remainingRepairs > 0) {
              state.remainingRepairs--;
              emit({ type: 'repair_attempted', correlationId, attempt: iteration });
              addTrace({ kind: 'repair', attempt: iteration, outcome: 'attempted' });

              const repair = await attemptRepair(
                model,
                iteration,
                `Validation failed: ${validation.failure.code} - ${validation.failure.message}`,
                snippet
              );
              if (repair.ok) {
                return {
                  ok: true,
                  turn: repair.turn,
                  trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
                };
              }
            }

            // Repair impossible or failed -> fall back to next model
            break;
          }

          // Success!
          emit({ type: 'validated', correlationId, state: validation.value.envelope.state, attempt: iteration });
          return {
            ok: true,
            turn: validation.value,
            trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
          };
        }

        // Model exhausted -> fall back to next model
        if (modelIdx < resolved.models.length - 1) {
          const nextModel = resolved.models[modelIdx + 1];
          emit({ type: 'fallback_model', correlationId, fromModel: model, toModel: nextModel });
          addTrace({ kind: 'fallback', fromModel: model, toModel: nextModel });
        }
      }

      // All models exhausted
      emit({ type: 'all_fallbacks_exhausted', correlationId });
      const failureContext: FailureContext = {
        correlationId,
        lastFailure: { code: 'malformed_json', message: 'All fallback models exhausted.' },
        trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
      };
      const fallbackTurn = (options.onFallbackExhausted ?? defaultFallbackHandler)(failureContext);
      return {
        ok: false,
        kind: 'all_fallbacks',
        turn: fallbackTurn,
        trace: { correlationId, entries: state.traceEntries, startedAt, endedAt: resolved.now() }
      };
    }
  };

  return { ok: true, harness };
}
