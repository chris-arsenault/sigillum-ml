# Whole-Score Composition Workflows for Sigillum ML

*Status: design research complete (2026-07-26); Ruby workflow and ML protocol
boundary implemented (2026-07-27). This document reviews Sigillum's agent
composition practice and historical sketch-to-score practices, then derives the
control and learning architecture for whole-score ML. The observed agent results
are medium quality, so the workflow is evidence about useful decomposition and
failure modes—not an expert demonstration of musical taste.*

## 1. Decision

Sigillum ML should follow the **control structure** of the agent practice, but
should not reproduce its literal sequence of stages.

The reusable control loop is:

```text
persistent score state
    -> select a musical scope, lens, and refinement operator
    -> generate several explicit score patches
    -> validate and judge locally, across seams, and globally
    -> accept, reject, retain the original, or backtrack
    -> update material dependencies and unresolved needs
```

This is a hierarchical blackboard process, not an eleven-stage state machine.
Partitura's Composition Graph and concrete composition snapshot are the
blackboard. Partitura Ruby owns the operational blackboard loop: scheduling,
candidate sandboxing, mechanical validation, promotion/rollback, and trajectory
continuity. Sigillum ML owns learned proposal generation, learned features and
critics, selection policies, and training.

The executable Ruby production source remains the musical authority. Accepted
candidate notes must be materialized there; model state, rejected candidates,
embeddings, critic outputs, preferences, and checkpoints remain external.

## 2. Sources and limits

### Sigillum sources

The review covered the current Partitura procedure and its superseded
predecessors:

- `../sigillum-library/reference/written/procedures/partitura/dsl_composition/`
- `../sigillum-library/reference/written/procedures/partitura/section_recomposition/`
- `../sigillum-library/reference/written/procedures/composition_method.md`
- `../sigillum-library/reference/written/procedures/unified_procedure.md`
- `../sigillum-library/reference/written/procedures/orchestration_procedure.md`
- `../sigillum-library/reference/written/procedures/post_merge_cypher_fractalization_procedure.md`
- `../sigillum-library/reference/written/procedures/section_repair_procedure.md`
- `../sigillum-library/docs/architecture/partitura/09_composition_graph.md`

Repository history was also inspected, especially:

- `9d6d7d9`: split a monolithic procedure into guided, stage-at-a-time runs;
- `d3422ea`: added surgical section recomposition;
- `b5dfc3c`: hardened the run against minimum-effort checklist completion;
- `b4f10c9`: repaired failures and friction found by field-test agents;
- `aa18b78`: clarified that local interleaving belongs inside span composition
  and the later pass is a global sweep for misses;
- `eb32957`: reduced pass notes to decisions, carries, improvements, and
  verdicts because realized music belongs in the score.

These procedures are not controlled composition experiments. Many rules were
added defensively after particular LLM failure modes. Their strongest evidence
is therefore about task decomposition, state handoff, and where errors
accumulate. Their medium-quality musical output is a warning against treating
their accepted note choices as ground-truth demonstrations.

### Historical sources

The historical comparison uses surviving manuscripts and institutional
descriptions:

- [Beethoven-Haus sketch archive](https://www.beethoven.de/en/archive/list/node/5179594084712448/Sketches)
  and [sketch glossary](https://www.beethoven.de/de/glossar_archiv/view/5378102137978880/Skizze)
- [Beethoven-Haus score-sketch glossary](https://www.beethoven.de/en/glossar_archiv/view/6212492447449088/Score%2Bsketch)
- [Library of Congress: Beethoven's *Das Schweigen*](https://www.loc.gov/collections/moldenhauer-archives/articles-and-essays/guide-to-archives/beethovens-das-schweigen/)
- [Neue Mozart Edition: Piano Concertos](https://dme.mozarteum.at/DME/objs/pdf/nma_157_-16_-3_eng.pdf)
- [Library of Congress: *Porgy and Bess*](https://www.loc.gov/collections/moldenhauer-archives/articles-and-essays/guide-to-archives/porgy-and-bess/)
- [Library of Congress: Aaron Copland Collection](https://www.loc.gov/collections/aaron-copland/about-this-collection/)
- [Partimenti overview](https://partimenti.org/overview/index.html)
  and [dispositions](https://partimenti.org/dispositions/about_dispo/overview.html)
- [University of Puget Sound: species counterpoint](https://musictheory.pugetsound.edu/mt21c/SpeciesCounterpoint.html)
- [IRCAM: Stravinsky's *Duo Concertant* process](https://ressources.ircam.fr/en/media/x8e96c4)
- [Library of Congress: Mahler sketches](https://www.loc.gov/collections/moldenhauer-archives/articles-and-essays/guide-to-archives/gustav-mahler-sketches/)
- [Mahler Foundation: Second Symphony manuscript](https://mahlerfoundation.org/mahler/compositions/symphony-no-2/symphony-no-2-manuscript/)

Surviving sketches are artifact- and preservation-biased. They expose some
externalized decisions, not the composer's complete mental process. Historical
practice supplies counterexamples and reusable structures, not a canonical
pipeline.

## 3. What the Sigillum workflow actually learned

The current procedure is an accumulated response to three recurring failures.

### 3.1 Local plausibility without purpose

One-shot writing can produce legal, locally plausible notes while failing to
establish a musical engine, phrase consequence, return identity, or destination.
The span loop counters this by:

- assigning a complete musical phrase or arc rather than an arbitrary token
  window;
- giving one pass ownership of all voices in that scope;
- establishing an active engine and its support;
- reading sounding projections;
- revising before moving to the next scope.

The important ML abstraction is **scope ownership plus a proposal/evaluation
loop**, not the fixed foreground-then-bass ordering.

### 3.2 Independent sections lose identity

The multi-agent orchestration procedure discovered that prose such as "return
of the opening motif" does not make independently written sections reproduce
the same musical object. It introduced named outputs containing exact material
and named downstream inputs. This turns fan-out into a dependency DAG.

Partitura's Composition Graph now provides the durable version of that idea:
stable material, phrase, placement, span, and section identities plus
`realizes`, `derives_from`, `returns_to`, and `depends_on` relations.

The ML controller should schedule against those identities. It should never
reconstruct a motif from an agent's description when the exact upstream
realization is available.

### 3.3 Good local blocks can still make a bad whole

Section agents cannot reliably hear:

- mechanical restarts at boundaries;
- loss of the whole-form destination;
- identical returns in new labels;
- aligned entrances and exits;
- accumulated density or register bias;
- long-range disappearance of a planned voice or material;
- dialect drift between otherwise adequate sections.

The merge and post-merge cypher passes exist for this blind spot. Their useful
abstraction is **critic scale separation**:

- local passage;
- neighboring seam;
- section;
- whole score;
- exported or rendered result.

A patch may improve one scale and damage another. Those evaluations must remain
separate long enough to expose the trade-off.

### 3.4 Fractalization is specificity, not density

The early orchestration procedure described detail as monotonically increasing.
Later revisions corrected the useful concept: refinement means a more specific
decision on the chosen dimension.

Examples include:

- a composed silence or reserve;
- a shortened tail;
- a pickup crossing a seam;
- a role exchange;
- a register hole;
- a brief doubling;
- a changed accompaniment rhythm;
- a clarified harmonic arrival;
- a resonance wake;
- removal of decorative activity.

The ML objective must not reward note count, changed-event count, or the fact
that a pass made an edit. The original score is always a candidate.

### 3.5 Validation is a floor

Compiler, range, graph-binding, and export checks are valuable hard filters.
They do not measure character, coherence, inevitability, memorability, pacing,
or taste. Sigillum's sounding projections are evidence available to critics;
declared intent is secondary bookkeeping.

This requires a lexicographic boundary:

1. reject mechanically invalid candidates;
2. compare valid candidates using plural musical judgments;
3. retain held-out human comparison outside the optimized critic ensemble.

## 4. What historical practice contributes

There is no single historical stage sequence to imitate.

Beethoven's sketches cover many resolutions, from short ideas through developed
drafts and score layouts. They support repeated transformation, but the
manuscript record does not justify forcing every work through uniform gradual
stages.

Mozart is an important counterexample to strict "composition, then
orchestration." Some concerto autographs acted as both draft and definitive
score: principal voices, bass, solo material, and salient wind ideas established
the structure before remaining parts were filled. Structural and timbral
identity could therefore coexist early in one evolving artifact.

Gershwin's complete short score for *Porgy and Bess* shows that
short-score-before-orchestration is also valid for some works. It should be an
available strategy, not a universal representation hierarchy.

Copland's surviving papers are not systematic, reinforcing that useful work may
occur out of order and at uneven resolution.

Partimento supplies a particularly useful ML analogy: a compact bass carries
constraints and affordances but admits multiple valid realizations. Species
counterpoint similarly demonstrates a curriculum of increasing constraint and
rhythmic complexity. Species is evidence for training operators progressively,
not evidence that the runtime controller must compose every score in species
order.

Stravinsky's surviving sources show rudimentary ideas developing across
surface, middleground, and large-scale documents. Mahler's revisions to
apparently complete orchestration show that a full or fair score is not an
irreversible terminal state.

The common design lesson is:

> Preserve multiple resolutions and stable identities, permit plural
> realizations, revisit earlier decisions, and keep stage boundaries porous.

## 5. Practices to retain, modify, and reject

| Agent practice | ML treatment |
|---|---|
| Hierarchical form, section, span, phrase scopes | Retain as stable graph-addressed scopes |
| Musical phrase/arc as the local work unit | Retain; allow smaller or larger scopes when the graph or critic requires them |
| One local pass owns all sounding voices | Retain as the default span-realization operator |
| Exact material carry-forward | Retain as native graph identity and dependency |
| Compile and sounding readouts after edits | Retain as hard validation plus critic evidence |
| Local, seam, merge, and post-export reviews | Retain as separate critic scales |
| Frozen unaffected music during repair | Retain as explicit patch scope and invariants |
| Eleven fixed stages | Replace with an event-driven scheduler and checkpoints |
| Fresh agents | Replace with independently conditioned policy/critic contexts; keep only if ablation shows benefit |
| Per-four-bar coverage | Replace with phrase-, risk-, uncertainty-, and critic-coverage scheduling |
| Foreground, bass, inner, support ordering | Use as an engine-first prior, not a law |
| Research commitments and technique cards | Use as retrieval-conditioned operator priors whose contribution is evaluated |
| Pass-note prose | Replace with structured transition records |
| Mandatory improvement every pass | Reject; compare against retaining the original |
| Monotonically increasing surface detail | Reject; optimize specificity for the musical job |
| A single aggregate quality score | Reject; preserve local, seam, global, and human signals |
| A fixed count of fresh contexts or revision passes | Reject |
| Uniform bar windows | Reject |

The ban on hidden generation applies to accepted score authority, not to model
internals. Models may generate candidates by any suitable method. Acceptance
means materializing the selected sounding result in Partitura.

## 6. Proposed composition kernel

### 6.1 Persistent state

The Ruby-owned composition state contains:

- a validated, immutable Partitura composition snapshot;
- its `graph_digest` and `snapshot_digest`;
- explicit unresolved refinement needs;
- an append-only trajectory of proposed actions and decisions.

The snapshot supplies stable paths, requirements, relations, concrete timed
events, and provenance. It is never an authoring format. Python receives that
state as an immutable protocol observation; it does not reconstruct a second
mutable score or workflow state.

### 6.2 Hierarchical controller

The operational Ruby controller selects:

```text
target graph path x musical lens x refinement operator
```

It does not emit notes directly. A learned controller may later recommend the
choice, but Partitura validates and records the executable action. Initial
lenses include form, primary material, harmony, rhythm, roles/texture, voice
leading, return identity, seam continuity, orchestration, pacing, idiom, and
audition.

Initial operators include planning, material establishment, span realization,
return transformation, vertical revision, seam interleaving, detailing,
subtraction/reserve, and targeted recomposition.

### 6.3 Candidate patches

An operator proposes one or more explicit patches against one exact snapshot
digest. Every patch records:

- candidate identity;
- base snapshot digest;
- target path;
- lens and operator;
- touched graph paths;
- content digest;
- either inline candidate patch text or an external artifact reference.

Partitura applies every patch only to an isolated temporary source, rebuilds the
snapshot, and optionally exports MusicXML and MIDI before selection. Python
never applies or promotes a patch. When a candidate wins, Partitura atomically
installs the exact bytes already validated in the sandbox, revalidates the
resulting snapshot, and rolls back on disagreement.

### 6.4 Critics

Separate evidence remains required for:

- Ruby-owned hard mechanical validation;
- local musical judgment;
- seam/neighbor judgment;
- sectional judgment;
- whole-score judgment;
- exported/rendered-result judgment.

Learned Python critics report named features and findings as well as optional
scores, but cannot claim the mechanical scale. Scores remain plural. A learned
policy may make a contextual decision, but the raw signals survive in the
Ruby-written trajectory.

### 6.5 Scheduler

The deterministic baseline scheduler should:

1. respect `depends_on` relations;
2. prefer completing partially bound requirements before unrelated open ones;
3. map requirement facets to appropriate lenses and operators;
4. accept explicit critic- or human-authored refinement needs for already bound
   nodes;
5. insert periodic whole-score reviews during refinement;
6. run final seam, form, vertical, orchestration, and audition reviews after
   mechanical binding;
7. stop after those review opportunities have been adjudicated, even if the
   decision was to keep the original.

This Ruby baseline is intentionally deterministic. It creates a behavioral
comparison point for a later learned scheduler. A learned scheduler remains an
advisor behind a future protocol revision; action validation and state
transition authority stay in Ruby.

## 7. Learning plan

The basic learning unit is a transition, not a complete score or an agent
transcript:

```text
snapshot_before
target path
musical lens
operator
candidate patches
per-candidate local/seam/global results
accepted patch, retained original, backtrack, or defer
snapshot_after when changed
updated requirements and unresolved needs
human comparison when available
```

### 7.1 How to use agent traces

Agent traces are weak supervision:

- imitate scope selection, dependency handling, validation, and tool use;
- learn operator priors from accepted and rejected attempts;
- do not assign expert labels to every accepted musical patch;
- retain failed candidates and "keep original" decisions;
- attach the known medium-quality status to whole trajectories.

Behavior cloning alone would inherit the agents' quality ceiling.

### 7.2 Training sequence

1. Instrument deterministic and agent-driven runs using the transition schema.
2. Train a controller warm start to predict target, lens, and operator.
3. Train local and global critics from pairwise candidate preferences,
   including the unedited original.
4. Measure critic agreement with held-out human comparison.
5. Train an offline contextual-bandit action recommender over existing
   trajectories.
6. Move to hierarchical offline RL only after reward diagnostics and
   off-policy evaluation are credible.
7. Keep a held-out human comparison set outside reward optimization.

The existing monophonic diffusion detailer is one candidate local-detail
operator. Its measured strength—adding surface detail to an established
line—does not make it the controller or a whole-score generator.

## 8. Evaluation

Compare at least:

1. one-shot whole-score generation;
2. a fixed agent-like stage pipeline;
3. the deterministic graph scheduler;
4. a learned adaptive action recommender behind Ruby validation.

Primary outcomes:

- blinded human pairwise preference;
- formal and thematic coherence;
- recognizable return and transformation identity;
- seam continuity;
- role and orchestration clarity;
- useful silence and reserve;
- mechanical validity;
- edit efficiency;
- critic/human agreement;
- diversity without identity loss.

Required ablations:

- no whole-score critic;
- no seam critic;
- no periodic global review;
- no exact material carry-forward;
- no candidate branching;
- no original-as-candidate;
- shared versus separated critic context;
- hand-authored versus learned features;
- post-export review disabled.

The central research question is not whether the system can produce more notes.
It is whether adaptive, graph-addressed refinement makes a whole score more
coherent and distinctive than both one-shot generation and the medium-quality
agent pipeline from which the control structure was derived.

### 8.1 Frozen evaluation lab

The implemented lab makes the comparison design executable without claiming
that the smoke fixture is research evidence.

- A versioned manifest pins every brief and starting Partitura source by
  SHA-256 and derives its own digest from canonical JSON.
- The manifest requires one-shot, fixed agent-like, and deterministic graph
  strategies, explicit seeds, all declared metric families, and the complete
  ablation set above.
- Every completed run is bound to one case/strategy/ablation/seed cell and a
  Partitura source digest. Duplicate cells are rejected.
- Edit efficiency is recorded from exact candidate, mechanically-valid
  candidate, accepted-edit, model-call, and wall-time counts. When a trajectory
  exists, candidate and acceptance counts are derived from it rather than
  entered manually.
- Partitura reports mechanical validity, requirement binding, explicit identity
  linkage, exact span-boundary behavior, texture/reserve descriptors, artifact
  digests, and event fingerprints. These are descriptive diagnostics, not
  coherence scores.
- Completed scores from any two systems can be exported by Partitura into a
  blinded A/B MusicXML/MIDI bundle. The private mapping records opaque
  evaluation run IDs; the public bundle does not.
- Evaluation preferences are always `held_out_evaluation`. The frozen
  comparison matrix includes all pairwise control baselines and each ablation
  against its corresponding control for every case, seed, and human criterion.
- Reports show missing run and comparison cells, plural system diagnostics,
  exact edit-effort summaries, fingerprint diversity, and per-criterion human
  wins/losses/ties/abstentions. They intentionally produce no composite reward,
  ranking, or automatic winner.

`experiments/whole_score/v1_smoke/manifest.json` is a contract smoke suite with
one case and one seed. It is expected to report `incomplete` until its 13 run
cells and 78 held-out comparisons are actually collected. Real claims require
larger frozen suites, independent raters, and an analysis plan; the repository
does not fabricate those results.

## 9. Implementation boundary

The implementation is split by capability, not by language convenience.

Partitura Ruby owns:

- graph and composition-snapshot construction;
- action, candidate, assessment, state, and transition invariants;
- deterministic dependency-aware scheduling and bounded execution;
- isolated patch application, compile, snapshot, and export;
- mechanical critic evidence;
- exact-byte promotion, concurrency checks, verification, and rollback;
- append-only trajectory, private review, and human-preference persistence;
- exact transition source/snapshot evidence and weak-supervision provenance;
- deterministic candidate replay and blinded A/B MusicXML/MIDI bundles;
- reusable completed-score measurement and blinded cross-system score review;
- versioned `proposal_request`, `proposal_response`, `selection_request`, and
  `selection_response` validation.

Sigillum ML Python contains only:

- immutable protocol DTOs;
- read-only trajectory, review, preference, and pairwise-training DTOs;
- frozen benchmark manifests, ML-run ledgers, held-out joins, aggregation, and
  report rendering;
- learned proposer, critic, selection-policy, and combined-provider interfaces;
- future learned implementations, training code, datasets, features, weights,
  checkpoints, and evaluation.

Python treats graph, snapshot, action, and candidate-evidence objects as opaque
protocol observations. It does not carry a graph implementation, score parser,
scheduler, executor, promoter, or trajectory state machine.

Trajectory schema v2 is independent from protocol schema v1. Every transition
contains the full pre-edit composition snapshot, exact UTF-8 Ruby source and its
digest, action, all candidate patches and critic evidence, decision, after
digests, unresolved paths, and a run context. A deterministic run must be
`unrated`; an agent-driven run must be explicitly labeled `medium`. This is a
trajectory-level statement of known data quality, not a reward.

Pairwise review is also Ruby-owned. `partitura review` reconstructs the exact
pre-edit score, replays a mechanically valid candidate, and writes anonymous A
and B MusicXML/MIDI files plus a public manifest that contains no candidate
identity or patch. A separate private JSONL record retains the blind mapping.
`partitura preference` records one A/B/tie/abstain judgment per review and marks
it either `training` or `held_out_evaluation`. Reusing one review in both
purposes is rejected.

## 10. Operationalization status

As of 2026-07-27, the corrected operational boundary is implemented:

- all repository tests execute with no skipped modules; the eight historical
  theme-generation modules no longer use a missing-package skip;
- ML-owned project paths and token-representation vocabulary remain local to
  this repository rather than recreating a general score framework;
- generated theme audition scores are materialized as explicit Ruby event
  lists and compiled/exported through the current Partitura CLI;
- Partitura Ruby implements scheduling, sandboxed candidate execution,
  MusicXML/MIDI capture, mechanical critic results, selection validation,
  exact-byte promotion and rollback, bounded runs, and durable trajectory
  persistence;
- `partitura observe`, `evaluate`, and `step` expose the process boundary as
  digest-bound JSON messages and keep the unchanged original explicit;
- Sigillum ML's `generation.composition` package now contains only those
  protocol DTOs, read-only evidence/training DTOs, and learned
  proposer/critic/policy interfaces;
- cross-repository behavioral tests prove Python proposals can be evaluated and
  selected while only Ruby mutates the accepted score and trajectory;
- transition records retain exact pre-edit source/snapshot evidence and every
  rejected candidate, with agent traces forced to carry their known
  medium-quality label;
- `partitura review` and `preference` implement a blinded, replay-verified
  human-comparison path with private identity mapping and separate training
  versus held-out dataset views.
- `partitura benchmark-score`, `benchmark-review`, and
  `benchmark-preference` measure completed sources and capture blinded
  cross-system evaluation without moving score interpretation into Python;
- the frozen evaluation lab verifies briefs and starting scores, requires the
  three baseline strategies and ten named ablations, rejects duplicate run
  cells, reports incomplete coverage honestly, and keeps human evaluation
  outside reward optimization.

The historical raw-MIDI dataset and categorization tools still name the removed
Python analysis API. They are not part of the active whole-score path and need a
separate migration onto a versioned Partitura analysis transport before they can
be treated as operational; restoring Python to `sigillum-library` is explicitly
not that migration.

Production learned proposal generation, musical critic implementations, a
justified best-of-K selection policy, corpus collection at useful scale, and
resumable model-service integration remain future operationalization phases.
The versioned evidence and preference-capture path is implemented; generated
corpora and review artifacts remain external experiment state rather than
repository content.
