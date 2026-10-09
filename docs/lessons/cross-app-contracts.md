# Lesson: Cross-App Contract Discipline

Lesson G from the cross-app AI convergence (2026-10). Three Applicaudia
apps (closedloop, strobopro, courtpuzzle) each evolved an AI stack from
the same ancestry; this library is the extraction and convergence point.
This file records the contract rules that keep a shared library honest
across hosts.

## One manifest per payload surface, states chosen per surface

A payload schema is declared once as a manifest + validator
(`createContract`), and the envelope states are a **per-surface decision**,
not a library constant. The same payload schema can back two contract
instances: a legacy single-shot surface with states `['proposal']` gets a
structural guarantee that a question envelope is invalid (`question` →
`envelope_state_invalid` → repair → still question → failure), while a chat
surface over the same schema adds `'question'` with conditional fields.
Never emulate "this surface must not ask questions" with prompt wording
alone — make the contract reject the state.

## Host seams: the failure contract (promoted from the closedloop seam)

Every host wraps `runTurn` in a thin seam (closedloop:
`functions/src/shared/ai/emcpHarness.ts` → `runPayloadTurn`). The seam
contract, learned the hard way:

- `result.ok === false` maps to a `kind: 'failed'` result — **never** to
  an assistant turn. The harness fallback question envelope is an internal
  last resort; rendering it as a model message teaches users to trust text
  no model wrote.
- Only the `question` kind carries `questionText`; `failed` has no text
  field at all. Type-level leak prevention: if the field doesn't exist,
  no consumer can accidentally render it.
- The host's consumer switch over result kinds is exhaustive
  (`assertNever` on the kind), so adding a kind (e.g. 0.3.0's
  `harness_threw` propagation) is a compile error at every call site, not
  a silent fall-through.
- Seams import shared constants (`RECOMMENDED_FLASH_MODEL_CHAIN`) and unit
  test that their default equals the library export — see
  [model-config.md](model-config.md). Re-declared constants drift;
  courtpuzzle shipped a real incident from a reader/writer path constant
  divergence before this rule existed.

## Validators: reject structural, clamp cosmetic

Payload validators separate two failure classes (the strobopro port
pattern): structural violations (note count out of range, missing
required fields) reject and route to repair — the model can fix those.
Cosmetic violations (a description one character over the limit) clamp to
the bound with a telemetry event carrying the field path — burning a
repair turn on a truncation is cost without benefit, and a silent clamp
without telemetry is unobservable drift. Never send an empty string to
the validator to "see what happens"; validate what arrived.

## Model chain history (promoted from the seam comments)

The chain constant exists because the hand-rolled lists kept going stale:
each app pinned whatever was current when its AI surface was written, and
retirements broke surfaces one at a time (a retired model as primary in
one app, an alias in a pinned slot in another). The library-level SSOT
plus seam-import-only adoption (above) is the structural fix. When a model
retires, the chain changes in one file and every app inherits it at its
next pin bump.

## Migration honesty (courtpuzzle, Phase 6)

Re-recorded record/replay fixtures prove self-consistency, **not**
behavior-unchanged. When migrating a provider onto the harness, the
behavior-unchanged claim rests on the mock lane and consumer-level tests
against unchanged consumer contracts, with fixtures as a smoke. Plan
migrations as staged commits (wire seam → swap surface → delete dead
plumbing) so each step's suite stays green.

Related: [harness-policy.md](harness-policy.md) (what the seam wraps),
[grounding-and-fencing.md](grounding-and-fencing.md) (what the seam's
prompt builder must fence).
