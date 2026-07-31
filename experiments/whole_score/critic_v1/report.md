# Pairwise critic readiness v1

This is the frozen phase-six critic-learning contract. It does not claim that a
critic has been trained.

The five critics are separate preference targets: coherence, material identity,
seams, orchestration, and reserve. Each label must come from a blinded
Partitura review that names exactly one criterion. Scale and criterion are
orthogonal: for example, a global review can judge coherence or reserve, while
the seam critic accepts only seam-scale comparisons.

The readiness gate requires, for every criterion:

- 200 training, 40 validation, and 80 held-out resolved comparisons;
- 8 training, 2 validation, and 4 held-out trajectory runs;
- both blinded A/B outcomes in every split;
- no run or candidate comparison shared with held-out evaluation;
- a usable difference between the Ruby-owned original and candidate snapshots.

Training preferences are assigned to training or validation by a frozen hash of
the trajectory run ID. Held-out preferences remain outside that split and
outside checkpoint selection.

The checked-in experiment intentionally has no expected corpus-index digest yet.
`python -m generation.tools.train_pairwise_critics audit` reports the missing
ignored index at
`outputs/datasets/whole_score/critic_v1/index.json`. Training refuses to start
until a real corpus exists, passes every gate, and its index digest is pinned in
the experiment contract.

The `index` subcommand content-addresses every trajectory, private review, and
preference JSONL file rather than trusting mutable paths. Any append or other
change requires rebuilding and repinning the index before training.

The model implementation is operational behind that gate: it constructs only
generic numeric and short-categorical differences from opaque Partitura
snapshots, fits train-only normalization, learns a shared representation with
five criterion-specific heads, balances pair orientation during training,
selects a checkpoint on validation balanced accuracy, evaluates the held-out
preferences once, and emits content-addressed artifacts. Python does not parse
scores or assign musical semantics to snapshot fields.

As of 2026-07-27 the measured corpus count is zero. Historical score annotations
remain representation supervision, not critic preferences; medium-quality
agent trajectories remain weak workflow evidence, not expert reward labels.
