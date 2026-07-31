# Whole-score ML boundary

This package contains only the Python side of Partitura's versioned ML protocol.
It does not implement a composition kernel or a second score runtime.

```text
Ruby Partitura observes and schedules
    -> ProposalRequest
Python learned proposer
    -> ProposalResponse with explicit source patches
Ruby sandboxes, compiles, exports, and mechanically evaluates
    -> SelectionRequest with immutable evidence and ephemeral score observations
Python learned critics and policy
    -> SelectionResponse (candidate or explicit original)
Ruby validates, promotes or retains, and appends the trajectory
    -> versioned transition evidence with exact before-source and all candidates
Ruby replays chosen variants and exports anonymous A/B MusicXML + MIDI
    -> private review mapping and human preference ledger
Python dataset readers
    -> disjoint training and held-out pairwise examples
```

- `protocol.py` provides immutable request/response DTOs and protocol validation.
- `interfaces.py` provides learned proposer, critic, policy, and combined-provider
  protocols.
- `evidence.py` provides immutable, read-only trajectory, private-review,
  preference, and pairwise-example DTOs. `training_pairs()` excludes
  `held_out_evaluation` records by construction.
- `benchmark.py`, `evaluation.py`, and `evaluation_store.py` provide frozen
  experiment manifests, completed-run records, held-out score-review joins,
  coverage checks, and plural reports. They consume Partitura measurements;
  they do not inspect or reinterpret score events.
- `observation_dataset.py` validates the pinned score-cohort specification,
  schedules Partitura projections, records annotation provenance, publishes an
  exact coverage manifest, and exposes read-only split views. It validates
  Ruby's observation transport without parsing MusicXML or deriving musical
  semantics in Python.
- `annotation_dataset.py` validates the versioned analytical-target
  specification, schedules Partitura annotation projections, gates known
  binding and audit outcomes, and exposes immutable target/split examples.
- `representation_baselines.py` fits train-only majority and standardized
  nearest-centroid models, rejects composition-lineage leakage, and reports
  validation/test accuracy and macro-F1 without creating music semantics.
- `score_span_encoder.py` prepares train-only feature normalization, trains one
  learned named-feature encoder with target-specific heads, selects a
  content-addressed checkpoint on validation macro-F1, performs one final test
  evaluation, and verifies artifact lineage. It learns analytical
  representations, not critic rewards.
- `structural_context.py` tensorizes canonical Partitura score observations,
  constructs same-score continuation supervision, learns ordered measure/span
  context, evaluates frozen checkpoints without fitting to external holdouts,
  and exposes the retained model as a threshold-free structural-seam signal.
  `StructuralSeamCritic` applies that signal to Ruby-exported candidate
  observations in a `SelectionRequest`. It does not parse MusicXML, select
  candidates, or claim general quality.
- `critic_learning.py` audits criterion-specific preference coverage, rejects
  held-out run/comparison leakage, constructs generic differences from opaque
  Partitura snapshots, and trains one shared representation with separate
  coherence, identity, seam, orchestration, and reserve heads. It refuses to
  train until the ignored corpus index is pinned and every frozen human-data
  gate passes.

Graph traversal, musical requirement scheduling, candidate execution, mechanical
validation, source promotion, rollback, and trajectory persistence all live in
`../sigillum-library` under Partitura's Ruby composition workflow. Python treats
the graph and score snapshot payloads as opaque observations. It may learn
features and weights from them, but it does not reinterpret them as an
independent musical data model. It also never replays patches, renders review
artifacts, assigns blind labels, or records human decisions.

Trajectory, review, preference, and rendered A/B files are generated corpus
state and must remain outside Git.

The phase-five lab is operated with
`python -m generation.tools.evaluate_composition`. Its checked-in smoke
manifest lives under `experiments/whole_score/`; it verifies the experiment
contract but does not contain or imply benchmark results.

The phase-six observation pilot is operated with
`python -m generation.tools.build_score_observations`. Its checked-in cohort
specification and measured coverage report live under `corpora/whole_score/`;
the 110 projected observations and exact generated manifest remain ignored.
The analytical-supervision slice is operated with
`python -m generation.tools.build_annotation_semantics`; its checked-in
semantic specification and measured baseline report live beside the cohort
specification. These examples are representation targets, not reward labels.
Criterion-specific critic targets require explicit candidate/original
comparisons and human preference evidence. Trajectory review/preference schema
v2 records criterion independently from review scale so those judgments are
actually trainable rather than inferred from prose.

The first learned representation pilot is operated with
`python -m generation.tools.train_score_span_encoder`. Its frozen specification
and durable measured report live under
`experiments/whole_score/representation_v1/`; checkpoints and machine-readable
reports remain ignored.

The label-balanced representation successor lives under
`experiments/whole_score/representation_v2/`. It is validation-only because the
v1 test split has already been exposed. Pairwise critic readiness and training
are operated with `python -m generation.tools.train_pairwise_critics`; the
frozen gates and measured zero-data status live under
`experiments/whole_score/critic_v1/`.

The structural-context sequence lives under
`experiments/whole_score/structural_context_v1/` through `v4/`. V4 is the first
version to improve over its controlled boundary baseline in all three
validation replications and on a separate external PDMX cohort. Its checkpoint
is callable through `StructuralSeamScorer`, but it supplies features only:
there is no learned pass/fail gate or composition policy. The
`StructuralSeamCritic` adapter emits those features for valid exported
workflow candidates and ignores candidates for which Ruby produced no score
observation.

The first real-candidate transfer probe is
`experiments/whole_score/seam_plausibility_v1/`. It sends 12 continuations
around current Movement IV excerpts (flute, clarinet, harp) through the real
Partitura workflow. The human batch compares the authentic continuation against
a boundary-matched plausible alternative, where the boundary baseline is tied on
all three items, so the choice isolates V4's learned residual (authentic once,
alternative twice); gross discontinuities are auto-rejected `3/3`. This is an
open plausibility question pending human ears, not a calibrated selector. The
same run builds three blinded General MIDI review bundles for that check.
It emits the generic review-manifest contract consumed by
`tools/sigillum-review-ui/`; later listening cadences reuse that tool rather
than adding cadence-specific pages or servers.

The rationale, historical review, training plan, and evaluation ablations are in
`docs/research/whole_score_composition_workflows.md`.
