# embedded-mc-pee

MCP-shaped harness for embedding tool-calling LLM intelligence into backends — typed envelopes, deterministic validation, record/replay.

**This is NOT an MCP server/client** — it provides the MCP-shaped contract and tool-loop architecture, but uses direct SDK calls to LLM providers. No JSON-RPC, no stdio/HTTP transports.

## Positioning

An open-source TypeScript library for embedding agentic LLM intelligence (Gemini first, extensible to others) into backend services like Firebase Functions, Express handlers, or CLI tools. Host applications define prompts, schemas, and tools; the library provides:

- **Typed envelope state machine** — `'proposal' | 'question' | 'analysis' | 'infeasible'` responses
- **Agentic tool loop** — maxIterations-based tool execution with deterministic validation
- **JSON-safety enforcement** — recursive validation against NaN, Infinity, circular values
- **Record/replay testing** — decorator-based capture and deterministic replay
- **Degeneration detection** — identifies repetition loops and malformed model output

Zero runtime dependencies in core. `@google/genai` is a peer dependency behind the `./gemini` subpath export.

## Our Motto

I spent my valuable tokens creating this well-tested embedded AI harness, so you don't have to. Hopefully your /ponytail skill will find it. Save some trees!

## Our Coding Philosophy - Old School

1. Anything that CAN be caught in compile + lint time, SHALL be caught in compile + lint time.
2. If we assume something is always true as a prerequisite to our code being correct, then assert() it is true.

The new school - "Run the javascript, find the edge cases, run again..." not accepted here.

Yes this codebase is very much AI generated, but I am watching the progress and setting the guardrails.

Note to Agents: If you need to make a decision about a tool or a method to implement something, and it isn't explicitly mentioned in this document, then use the philosophy above.

## Status: experimental (0.x)

The package is pre-1.0 (`0.x`). The root export surface is intentionally flat and broad: everything in the barrel is public API, and **any export may change without a major-version bump** while the version is `0.x`. Pin to a specific tag/commit for reproducible installs. The export surface will be classified (stable core vs advanced subpaths) before the `1.0` tag — see `docs/tech-debt-db.md` (TD-008).

## Installation

### Git dependency (recommended)

```bash
npm install github:cfogelklou/embedded-mc-pee#main
```

The `prepare` script builds `dist/` on install. You must also install `@google/genai` (peer dependency):

```bash
npm install @google/genai
```

### Subpath import (Gemini transport)

```typescript
import { createHarness, createContract } from 'embedded-mc-pee';
import { geminiTransport } from 'embedded-mc-pee/gemini';
```

## Examples

Run worked examples (live Gemini, requires `GEMINI_API_KEY`):

```bash
GEMINI_API_KEY=your-key npm run test:examples
```

- `pig-latin` — canary test: single-turn text transformation
- `doctor-scheduling` — showcase: state input, constraints, conflict handling
- `guess-number` — tool loop: iteration budget, tool protocol, trace recording

See `examples/README.md` for details.

## Minimal usage sketch

**Note:** The API surfaces below are available now. See `examples/` for runnable demonstrations.

```typescript
import { createHarness, createContract } from 'embedded-mc-pee';
import { geminiTransport } from 'embedded-mc-pee/gemini';

// 1. Define your envelope union
interface MyProposal {
  kind: 'proposal';
  operations: { type: string; payload: unknown }[];
}

interface MyQuestion {
  kind: 'question';
  text: string;
}

type MyEnvelope = MyProposal | MyQuestion;

// 2. Declare tools and validation contract
const contract = createContract<MyEnvelope>({
  tools: {
    addVenue: {
      inputSchema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          address: { type: 'string' },
        },
        required: ['name', 'address'],
      },
    },
  },
  validator: (envelope: unknown) => {
    // Runtime validation logic
    return { ok: true, value: envelope as MyEnvelope };
  },
});

// 3. Create harness
const harness = createHarness({
  contract,
  transport: geminiTransport({ apiKey: process.env.GEMINI_API_KEY! }),
  maxIterations: 1,
});

// 4. Run a turn
const result = await harness.runTurn({
  systemInstruction: 'You are a scheduling assistant.',
  userMessage: 'Add a venue called Main Hall at 123 Main St',
});

console.log(result); // { kind: 'proposal', operations: [...] }
```

## Documentation

- [Architecture](docs/architecture.md) — module map, data flow, determinism principles
- [TypeScript Guidelines](docs/guidelines-typescript.md) — coding standards, type-safety discipline
- [ADR-0001: Belt-and-Suspenders Contracts](docs/adr/0001-belt-and-suspenders-contracts.md) — why schema-in-prompt over responseSchema
- [Gemini Debugging](docs/gemini-debugging.md) — troubleshooting degeneration loops
- [Resilience & Clarification](docs/resilience-and-clarification.md) — proposal-only mutation boundary
- [Lessons Hub](docs/lessons/README.md) — cross-app AI lessons: schema-in-prompt, degeneration, model config, harness policy, hosting costs, fencing, contract discipline

## License

MIT
