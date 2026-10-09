/**
 * Tests for record/recordReplay.ts
 *
 * Recording, replay, serialization, and end-to-end harness tests.
 */

import { describe, it, expect } from 'vitest';
import type { LlmTransport, LlmRequest, LlmResponse } from '../transport/types';
import {
  RecordingTransport,
  ReplayTransport,
  ReplayMissError,
  serializeRecording,
  parseRecording,
  type RecordingEntries,
  type OnRecordCallback
} from './recordReplay';
import { createHarness } from '../policy/executor';
import { createContract } from '../contract/contract';
import { DEFAULT_REPLAY_RECORDING_VERSION } from './hash';

// ============================================================================
// Test Doubles
// ============================================================================

/**
 * Fake transport that returns predefined responses.
 */
class FakeTransport implements LlmTransport {
  private readonly responses: Array<LlmResponse | Error>;
  private callCount = 0;

  constructor(responses: Array<LlmResponse | Error>) {
    this.responses = responses;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async complete(_req: LlmRequest, _opts: { timeoutMs: number }): Promise<LlmResponse> {
    const response = this.responses[this.callCount];
    this.callCount++;

    if (response instanceof Error) {
      throw response;
    }

    return response;
  }

  getCallCount(): number {
    return this.callCount;
  }
}

// ============================================================================
// Test Data
// ============================================================================

const makeRequest = (overrides?: Partial<LlmRequest>): LlmRequest => ({
  model: 'gemini-2.0-flash',
  systemInstruction: 'You are a helpful assistant.',
  promptText: 'What is 2+2?',
  toolDeclarations: [],
  ...overrides
});

const makeResponse = (overrides?: Partial<LlmResponse>): LlmResponse => ({
  rawText: JSON.stringify({ answer: '4' }),
  finishReason: 'STOP',
  usage: { outputTokens: 10 },
  ...overrides
});

// ============================================================================
// Recording Transport Tests
// ============================================================================

describe('RecordingTransport', () => {
  it('should record successful responses from inner transport', async () => {
    const response = makeResponse({ rawText: 'Success!' });
    const transport = new FakeTransport([response]);
    const entries = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(transport, entries);

    const req = makeRequest();
    const result = await recorder.complete(req, { timeoutMs: 5000 });

    expect(result).toEqual(response);
    expect(entries.size).toBe(1);

    const entry = Array.from(entries.values())[0];
    expect(entry.request).toEqual(req);
    expect(entry.response).toEqual(response);
  });

  it('should record multiple responses', async () => {
    const response1 = makeResponse({ rawText: 'First' });
    const response2 = makeResponse({ rawText: 'Second' });
    const transport = new FakeTransport([response1, response2]);
    const entries = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(transport, entries);

    const req1 = makeRequest({ promptText: 'First request' });
    const req2 = makeRequest({ promptText: 'Second request' });

    await recorder.complete(req1, { timeoutMs: 5000 });
    await recorder.complete(req2, { timeoutMs: 5000 });

    expect(entries.size).toBe(2);
    expect(transport.getCallCount()).toBe(2);
  });

  it('should NOT record rejections from inner transport', async () => {
    const error = new Error('Network failure');
    const transport = new FakeTransport([error]);
    const entries = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(transport, entries);

    const req = makeRequest();

    await expect(recorder.complete(req, { timeoutMs: 5000 })).rejects.toThrow('Network failure');
    expect(entries.size).toBe(0);
  });

  it('should invoke onRecord callback after successful recording', async () => {
    const response = makeResponse({ rawText: 'Callback test' });
    const transport = new FakeTransport([response]);
    const entries = new Map<string, { request: LlmRequest; response: LlmResponse }>();

    let callbackInvoked = false;
    let capturedHash: string | undefined;
    let capturedRequest: LlmRequest | undefined;
    let capturedResponse: LlmResponse | undefined;

    const onRecord: OnRecordCallback = (hash, request, resp) => {
      callbackInvoked = true;
      capturedHash = hash;
      capturedRequest = request;
      capturedResponse = resp;
    };

    const recorder = new RecordingTransport(transport, entries, onRecord);
    const req = makeRequest();

    await recorder.complete(req, { timeoutMs: 5000 });

    expect(callbackInvoked).toBe(true);
    expect(capturedHash).toBeDefined();
    expect(capturedRequest).toEqual(req);
    expect(capturedResponse).toEqual(response);
  });

  it('should preserve RAW LlmResponse verbatim including toolCall and usage', async () => {
    const response: LlmResponse = {
      rawText: 'Tool result',
      toolCall: { name: 'calculator', args: { expression: '2+2' } },
      finishReason: 'STOP',
      usage: { outputTokens: 15, thoughtsTokenCount: 100 }
    };

    const transport = new FakeTransport([response]);
    const entries = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(transport, entries);

    const req = makeRequest();
    await recorder.complete(req, { timeoutMs: 5000 });

    const recorded = Array.from(entries.values())[0].response;
    expect(recorded).toEqual(response);
    expect(recorded.toolCall).toEqual(response.toolCall);
    expect(recorded.usage).toEqual(response.usage);
    expect(recorded.finishReason).toEqual(response.finishReason);
  });
});

// ============================================================================
// Replay Transport Tests
// ============================================================================

describe('ReplayTransport', () => {
  it('should return byte-identical responses for recorded requests', async () => {
    const response = makeResponse({ rawText: 'Replayed response' });

    // Need to use real hash, so we'll create real entries via recording
    const fakeTransport = new FakeTransport([response]);
    const realEntries = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(fakeTransport, realEntries);

    const recordedReq = makeRequest();
    await recorder.complete(recordedReq, { timeoutMs: 5000 });

    const recordedEntries = Object.fromEntries(realEntries) as RecordingEntries;
    const replay = new ReplayTransport(recordedEntries);

    const replayedResponse = await replay.complete(recordedReq, { timeoutMs: 5000 });

    expect(replayedResponse).toEqual(response);
    expect(JSON.stringify(replayedResponse)).toBe(JSON.stringify(response));
  });

  it('should reject with ReplayMissError in strict mode on hash miss', async () => {
    const entries: RecordingEntries = {};
    const replay = new ReplayTransport(entries, { missMode: 'strict' });

    const req = makeRequest({ promptText: 'Unrecorded request' });

    await expect(replay.complete(req, { timeoutMs: 5000 })).rejects.toThrow();

    try {
      await replay.complete(req, { timeoutMs: 5000 });
      expect.fail('Should have thrown');
    } catch (err: unknown) {
      const error = err as ReplayMissError;
      expect(error.kind).toBe('REPLAY_MISS');
      expect(error.hash).toBeDefined();
      expect(error.requestSnippet).toBeDefined();
      expect(error.message).toContain('Replay miss');
    }
  });

  it('should include request snippet in ReplayMissError', async () => {
    const entries: RecordingEntries = {};
    const replay = new ReplayTransport(entries, { missMode: 'strict' });

    const longPrompt = 'A'.repeat(200);
    const req = makeRequest({ promptText: longPrompt });

    try {
      await replay.complete(req, { timeoutMs: 5000 });
      expect.fail('Should have thrown');
    } catch (err: unknown) {
      const error = err as ReplayMissError;
      expect(error.requestSnippet).toBeDefined();
      expect(error.requestSnippet.length).toBeLessThanOrEqual(100);
    }
  });

  it('should forward to inner transport in passthrough mode on miss', async () => {
    const response = makeResponse({ rawText: 'Passthrough response' });
    const innerTransport = new FakeTransport([response]);
    const entries: RecordingEntries = {};
    const replay = new ReplayTransport(entries, { missMode: 'passthrough' }, innerTransport);

    const req = makeRequest({ promptText: 'Unrecorded request' });
    const result = await replay.complete(req, { timeoutMs: 5000 });

    expect(result).toEqual(response);
    expect(innerTransport.getCallCount()).toBe(1);
  });

  it('should reject passthrough mode without inner transport', async () => {
    const entries: RecordingEntries = {};
    const replay = new ReplayTransport(entries, { missMode: 'passthrough' });

    const req = makeRequest({ promptText: 'Unrecorded request' });

    await expect(replay.complete(req, { timeoutMs: 5000 })).rejects.toThrow('Passthrough mode requires inner transport');
  });

  it('should default to strict mode', async () => {
    const entries: RecordingEntries = {};
    const replay = new ReplayTransport(entries); // No missMode specified

    const req = makeRequest({ promptText: 'Unrecorded request' });

    try {
      await replay.complete(req, { timeoutMs: 5000 });
      expect.fail('Should have thrown');
    } catch (err: unknown) {
      const error = err as ReplayMissError;
      expect(error.kind).toBe('REPLAY_MISS');
    }
  });
});

// ============================================================================
// Serialization Tests
// ============================================================================

describe('serializeRecording', () => {
  it('should serialize entries to JSON string', () => {
    const req = makeRequest();
    const response = makeResponse();
    const entries: RecordingEntries = {
      ['hash1']: { request: req, response }
    };

    const result = serializeRecording(entries);

    expect(result.ok).toBe(true);
    if (result.ok) {
      const parsed = JSON.parse(result.value);
      expect(parsed.recordingVersion).toBe(DEFAULT_REPLAY_RECORDING_VERSION);
      expect(parsed.entries).toBeDefined();
    }
  });

  it('should include recordingVersion in output', () => {
    const entries: RecordingEntries = {};
    const result = serializeRecording(entries);

    expect(result.ok).toBe(true);
    if (result.ok) {
      const parsed = JSON.parse(result.value);
      expect(parsed.recordingVersion).toBe(DEFAULT_REPLAY_RECORDING_VERSION);
    }
  });

  it('should handle complex tool calls in serialization', () => {
    const req = makeRequest();
    const response: LlmResponse = {
      rawText: 'Complex response',
      toolCall: {
        name: 'complex_tool',
        args: {
          nested: { data: [1, 2, 3] },
          flag: true
        }
      },
      finishReason: 'STOP',
      usage: { outputTokens: 42 }
    };

    const entries: RecordingEntries = {
      ['hash1']: { request: req, response }
    };

    const result = serializeRecording(entries);
    expect(result.ok).toBe(true);
  });
});

describe('parseRecording', () => {
  it('should parse valid recording JSON', () => {
    const req = makeRequest();
    const response = makeResponse();
    const entries: RecordingEntries = {
      ['hash1']: { request: req, response }
    };

    const serialized = serializeRecording(entries);
    expect(serialized.ok).toBe(true);

    if (serialized.ok) {
      const parsed = parseRecording(serialized.value);
      expect(parsed.ok).toBe(true);
      if (parsed.ok) {
        expect(parsed.entries).toEqual(entries);
      }
    }
  });

  it('should reject old recordingVersion with typed failure including observed version', () => {
    const oldRecording = JSON.stringify({
      recordingVersion: 0, // Old version
      entries: {}
    });

    const result = parseRecording(oldRecording);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('WRONG_VERSION');
      expect(result.failure.observedVersion).toBe(0);
      expect(result.failure.reason).toContain('Unsupported recording version');
    }
  });

  it('should reject future recordingVersion with typed failure', () => {
    const futureRecording = JSON.stringify({
      recordingVersion: DEFAULT_REPLAY_RECORDING_VERSION + 1,
      entries: {}
    });

    const result = parseRecording(futureRecording);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('WRONG_VERSION');
      expect(result.failure.observedVersion).toBe(DEFAULT_REPLAY_RECORDING_VERSION + 1);
    }
  });

  it('should reject corrupt JSON with typed failure', () => {
    const corruptJson = '{ invalid json }';

    const result = parseRecording(corruptJson);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('INVALID_JSON');
      expect(result.failure.reason).toContain('JSON.parse failed');
    }
  });

  it('should reject unknown top-level keys with typed failure', () => {
    const recordingWithExtraKey = JSON.stringify({
      recordingVersion: DEFAULT_REPLAY_RECORDING_VERSION,
      entries: {},
      unknownKey: 'should not be here'
    });

    const result = parseRecording(recordingWithExtraKey);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('UNKNOWN_KEY');
      expect(result.failure.reason).toContain('Unknown top-level keys');
    }
  });

  it('should reject missing recordingVersion with typed failure', () => {
    const recordingWithoutVersion = JSON.stringify({
      entries: {}
    });

    const result = parseRecording(recordingWithoutVersion);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('MISSING_VERSION');
      expect(result.failure.reason).toContain('Missing recordingVersion');
    }
  });

  it('should reject missing entries with typed failure', () => {
    const recordingWithoutEntries = JSON.stringify({
      recordingVersion: DEFAULT_REPLAY_RECORDING_VERSION
    });

    const result = parseRecording(recordingWithoutEntries);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('MISSING_ENTRIES');
      expect(result.failure.reason).toContain('Missing entries');
    }
  });

  it('should serialize→parse roundtrip identically', () => {
    const req1 = makeRequest({ promptText: 'First' });
    const response1 = makeResponse({ rawText: 'Response 1' });
    const req2 = makeRequest({ promptText: 'Second' });
    const response2 = makeResponse({ rawText: 'Response 2' });

    const originalEntries: RecordingEntries = {
      ['hash1']: { request: req1, response: response1 },
      ['hash2']: { request: req2, response: response2 }
    };

    const serialized = serializeRecording(originalEntries);
    expect(serialized.ok).toBe(true);

    if (serialized.ok) {
      const parsed = parseRecording(serialized.value);
      expect(parsed.ok).toBe(true);

      if (parsed.ok) {
        expect(parsed.entries).toEqual(originalEntries);
      }
    }
  });
});

// ============================================================================
// End-to-End Tests
// ============================================================================

describe('End-to-end with createHarness', () => {
  it('should replay recorded success through harness pipeline', async () => {
    // First, record a successful interaction
    const recordedResponse: LlmResponse = {
      rawText: JSON.stringify({
        state: 'question',
        questionText: 'What is your name?'
      }),
      finishReason: 'STOP'
    };

    const fakeTransport = new FakeTransport([recordedResponse]);
    const recordingMap = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(fakeTransport, recordingMap);

    // Create a simple contract for question state
    const contractResult = createContract({
      envelope: {
        states: ['question', 'proposal', 'infeasible', 'analysis']
      },
      payload: {
        type: 'object',
        properties: {},
        required: []
      }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    }, (_payload: unknown) => {
      // Simple validator that accepts any payload
      return { ok: true, value: {} };
    });

    expect(contractResult.ok).toBe(true);
    if (!contractResult.ok) {
      throw new Error('Failed to create contract');
    }

    const contract = contractResult.contract;

    const harnessResult = createHarness({
      transport: recorder,
      models: ['gemini-2.0-flash'],
      maxIterations: 1
    }, contract);

    expect(harnessResult.ok).toBe(true);
    if (!harnessResult.ok) {
      throw new Error('Failed to create harness');
    }

    const harness = harnessResult.harness;

    // Run a turn to record the interaction
    const recordResult = await harness.runTurn({
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'Ask me a question.'
    });

    expect(recordResult.ok).toBe(true);
    expect(recordResult.turn.envelope.state).toBe('question');

    // Now create a replay harness with the recorded entries
    const recordedEntries = Object.fromEntries(recordingMap) as RecordingEntries;
    const replayTransport = new ReplayTransport(recordedEntries, { missMode: 'strict' });

    const replayHarnessResult = createHarness({
      transport: replayTransport,
      models: ['gemini-2.0-flash'],
      maxIterations: 1
    }, contract);

    expect(replayHarnessResult.ok).toBe(true);
    if (!replayHarnessResult.ok) {
      throw new Error('Failed to create replay harness');
    }

    const replayHarness = replayHarnessResult.harness;

    // Run the same turn with replay - should get identical result
    const replayResult = await replayHarness.runTurn({
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'Ask me a question.'
    });

    expect(replayResult.ok).toBe(true);
    expect(replayResult.turn.envelope.state).toBe('question');
    expect(replayResult.turn.envelope.questionText).toBe('What is your name?');
  });

  it('should handle tool call replays through harness', async () => {
    // Record a tool call response
    const recordedResponse: LlmResponse = {
      rawText: 'I will help with that.',
      toolCall: {
        name: 'calculator',
        args: { expression: '2+2' }
      },
      finishReason: 'STOP'
    };

    const fakeTransport = new FakeTransport([recordedResponse]);
    const recordingMap = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(fakeTransport, recordingMap);

    // Create a contract with tool support
    const contractResult = createContract({
      envelope: {
        states: ['question', 'proposal', 'infeasible', 'analysis']
      },
      payload: {
        type: 'object',
        properties: {},
        required: []
      }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    }, (_payload: unknown) => {
      // Simple validator that accepts any payload
      return { ok: true, value: {} };
    });

    expect(contractResult.ok).toBe(true);
    if (!contractResult.ok) {
      throw new Error('Failed to create contract');
    }

    const contract = contractResult.contract;

    const harnessResult = createHarness({
      transport: recorder,
      models: ['gemini-2.0-flash'],
      maxIterations: 1,
      tools: [
        {
          contract: {
            name: 'calculator',
            inputSchema: {
              type: 'object',
              properties: {
                expression: { type: 'string' }
              },
              required: ['expression']
            }
          },
          handler: async () => ({
            text: 'Result: 4'
          })
        }
      ]
    }, contract);

    expect(harnessResult.ok).toBe(true);
    if (!harnessResult.ok) {
      throw new Error('Failed to create harness');
    }

    const harness = harnessResult.harness;

    // Record the interaction
    await harness.runTurn({
      systemInstruction: 'You are a helpful assistant.',
      promptText: 'Calculate 2+2'
    });

    // The interaction should have been recorded
    expect(recordingMap.size).toBeGreaterThan(0);

    // Verify the recording captured the tool call
    const recordedEntries = Object.fromEntries(recordingMap) as RecordingEntries;
    const firstEntry = Object.values(recordedEntries)[0];
    expect(firstEntry.response.toolCall).toEqual({
      name: 'calculator',
      args: { expression: '2+2' }
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Recording + Replay Integration', () => {
  it('should record and replay complex interaction with multiple fields', async () => {
    // Create a response with all fields populated
    const complexResponse: LlmResponse = {
      rawText: JSON.stringify({
        state: 'proposal',
        payload: { result: 42, calculation: '2+2' }
      }),
      finishReason: 'STOP',
      usage: { outputTokens: 25, thoughtsTokenCount: 150 }
    };

    const fakeTransport = new FakeTransport([complexResponse]);
    const recordingMap = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(fakeTransport, recordingMap);

    const req = makeRequest({
      promptText: 'Calculate 2+2',
      temperature: 0.5,
      maxOutputTokens: 1000
    });

    await recorder.complete(req, { timeoutMs: 5000 });

    expect(recordingMap.size).toBe(1);

    // Replay the recorded interaction
    const recordedEntries = Object.fromEntries(recordingMap) as RecordingEntries;
    const replay = new ReplayTransport(recordedEntries, { missMode: 'strict' });

    const replayedResponse = await replay.complete(req, { timeoutMs: 5000 });

    expect(replayedResponse).toEqual(complexResponse);
    expect(replayedResponse.usage).toEqual(complexResponse.usage);
    expect(replayedResponse.finishReason).toBe(complexResponse.finishReason);
  });

  it('should handle recording with optional request fields', async () => {
    const response1 = makeResponse({ rawText: 'Response for req1' });
    const response2 = makeResponse({ rawText: 'Response for req2' });
    const response3 = makeResponse({ rawText: 'Response for req3' });
    const fakeTransport = new FakeTransport([response1, response2, response3]);
    const recordingMap = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(fakeTransport, recordingMap);

    // Request with all optional fields
    const req1 = makeRequest({
      promptText: 'Request with all optional fields',
      temperature: 0.7,
      maxOutputTokens: 2048,
      thinkingLevel: 'medium'
    });

    await recorder.complete(req1, { timeoutMs: 5000 });

    // Request with different optional fields
    const req2 = makeRequest({
      promptText: 'Request with temperature only',
      temperature: 0.3
    });

    await recorder.complete(req2, { timeoutMs: 5000 });

    // Request with no optional fields
    const req3 = makeRequest({
      promptText: 'Request with no optional fields'
    });

    await recorder.complete(req3, { timeoutMs: 5000 });

    expect(recordingMap.size).toBe(3);

    // All should replay correctly
    const recordedEntries = Object.fromEntries(recordingMap) as RecordingEntries;
    const replay = new ReplayTransport(recordedEntries, { missMode: 'strict' });

    const replay1 = await replay.complete(req1, { timeoutMs: 5000 });
    const replay2 = await replay.complete(req2, { timeoutMs: 5000 });
    const replay3 = await replay.complete(req3, { timeoutMs: 5000 });

    expect(replay1.rawText).toBe(response1.rawText);
    expect(replay2.rawText).toBe(response2.rawText);
    expect(replay3.rawText).toBe(response3.rawText);
  });

  it('should replay a hit regardless of temperature on the request (v2 hash)', async () => {
    const response = makeResponse({ rawText: 'Response for temperature-independent request' });
    const fakeTransport = new FakeTransport([response]);
    const recordingMap = new Map<string, { request: LlmRequest; response: LlmResponse }>();
    const recorder = new RecordingTransport(fakeTransport, recordingMap);

    // Record WITH a (deprecated, never-transmitted) temperature...
    await recorder.complete(
      makeRequest({ promptText: 'Temperature-independent request', temperature: 0.7 }),
      { timeoutMs: 5000 }
    );

    const recordedEntries = Object.fromEntries(recordingMap) as RecordingEntries;
    // ...strict-mode replay WITHOUT it must still hit the same entry.
    const replay = new ReplayTransport(recordedEntries, { missMode: 'strict' });
    const replayed = await replay.complete(
      makeRequest({ promptText: 'Temperature-independent request' }),
      { timeoutMs: 5000 }
    );

    expect(replayed.rawText).toBe(response.rawText);
  });
});
