# Whole-score ML boundary

This package contains only the Python side of Partitura's versioned ML protocol.
It does not implement a composition kernel or a second score runtime.

```text
Ruby Partitura observes and schedules
    -> ProposalRequest
Python learned proposer
    -> ProposalResponse with explicit source patches
Ruby sandboxes, compiles, exports, and mechanically evaluates
    -> SelectionRequest with immutable candidate evidence
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
Criterion-specific critic targets still require explicit candidate/original
comparisons and human preference evidence.

The rationale, historical review, training plan, and evaluation ablations are in
`docs/research/whole_score_composition_workflows.md`.
