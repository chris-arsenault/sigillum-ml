# Whole-score annotation semantics pilot v1

This report records the first complete build and baseline measurement of
`whole_score_annotation_semantics_pilot_v1`. It is a durable summary of the
pinned specification in `annotation_semantics_v1.json`; generated projections,
manifests, and baseline files remain outside Git.

## Boundary

The slice makes historical analytical annotations operational for
representation learning without turning them into composition rewards:

- Ruby Partitura is the sole authority for MusicXML/MXL interpretation,
  external-annotation semantics, canonical score binding, and factual feature
  extraction.
- Python schedules projections, verifies parent and source digests, enforces
  composition-lineage splits, exposes immutable training views, and fits
  train-only measurement baselines.
- No Python score parser or second musical intermediate representation is
  introduced.
- Form, harmony, orchestral-role, Hauptstimme, relation, recurrence, and
  boundary labels describe score facts. They do not judge whether a generated
  edit is good.
- Candidate/original changes and the five criterion-specific critics remain
  explicitly unavailable until composition trajectories and blind human
  comparisons exist.

## Measured projection

The complete projector-revision `partitura-annotation-observation-v1-r1`
build finished on 2026-07-27 with every declared coverage gate passing:

| Measurement | Count |
| --- | ---: |
| Scores projected | 110 |
| Failed scores | 0 |
| Binding failures | 0 |
| Known failed alignment audits | 10 |
| Supported targets populated | 8 |
| Explicitly unavailable targets | 6 |
| Training examples | 124,146 |
| Validation examples | 42,687 |
| Test examples | 63,451 |
| **Total examples** | **230,284** |
| Row-addressed warnings | 7,754 |

The ten gated audit mismatches are two OpenScore movements whose derived
part-relation rows fall wholly outside the unexpanded score timeline, six S3
downbeat timelines that disagree with Partitura measure offsets, and two S3
time-signature tables that disagree with Partitura meter events. These do not
invalidate other bound examples from the affected score. They are part of the
expected measured state, not hidden success.

Target coverage is:

| Target | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Structural part relation | 71,957 | 26,177 | 45,601 | 143,735 |
| Prominent part | 6,351 | 1,498 | 2,008 | 9,857 |
| Material recurrence | 6,525 | 1,562 | 2,183 | 10,270 |
| Form section | 253 | 88 | 190 | 531 |
| Cadence type | 217 | 58 | 118 | 393 |
| Harmonic function | 3,149 | 1,408 | 2,718 | 7,275 |
| Orchestral role | 7,480 | 3,694 | 9,243 | 20,417 |
| Seam boundary | 23,214 | 6,202 | 8,390 | 37,806 |

## Source-quality accounting

Warnings retain exact source path and row provenance. Their distribution is:

| Warning | Count | Interpretation |
| --- | ---: | --- |
| `numbered_annotation_bound_to_combined_part` | 3,986 | S3 distinguishes players that the score encodes as one part. |
| `ambiguous_part_resolved_by_activity` | 2,317 | A source name matched multiple score parts; span-local activity selected one. |
| `annotation_span_outside_score` | 1,302 | Usually repeat-expanded OpenScore relation tails outside the unexpanded score. |
| `invalid_annotation_span` | 54 | Reversed or zero-length source span; no example was invented. |
| `annotation_span_clamped` | 30 | A partly intersecting source span was bounded to the score timeline. |
| `invalid_numeric_annotation` | 27 | Malformed or concatenated source row lacked usable numeric coordinates. |
| `combined_part_resolved_by_activity` | 24 | A numbered player within a multi-player score part was selected by activity. |
| `s3_temporal_audit_failed` | 8 | Six downbeat and two meter-table alignment audits failed. |
| `missing_material_label` | 6 | Form supervision was retained, while recurrence supervision was withheld. |

Warnings are not binding failures. Each code describes a declared
interpretation or an excluded row; no malformed row is silently converted into
a label.

## Non-neural baselines

Both models fit only the training split. The centroid baseline standardizes
features with train-only mean and variance, computes one centroid per training
label, and never observes validation or test labels while fitting. All three
pairwise lineage-overlap checks passed with empty overlap.

| Target | Labels | Features | Majority test macro-F1 | Centroid validation macro-F1 | Centroid test macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structural part relation | 8 | 108 | 0.089157 | 0.201775 | 0.190021 |
| Prominent part | 101 | 36 | 0.007240 | 0.013789 | 0.013078 |
| Material recurrence | 2 | 18 | 0.447901 | 0.586081 | 0.626852 |
| Form section | 27 | 18 | 0.007702 | 0.055026 | 0.003205 |
| Cadence type | 4 | 18 | 0.295597 | 0.178289 | 0.246101 |
| Harmonic function | 265 | 18 | 0.003114 | 0.000262 | 0.002712 |
| Orchestral role | 9 | 36 | 0.050721 | 0.101780 | 0.119862 |
| Seam boundary | 2 | 54 | 0.425224 | 0.524985 | 0.518602 |

The factual span features materially improve macro-F1 over majority for held-out
relations, recurrence, orchestral role, and seams. They do not solve the
source-local, high-cardinality label spaces for form, harmony, or specific part
names. Prominent-part evaluation also contains unseen instrument labels in
validation and test. Learned representation work should therefore preserve
the successful relational/boundary signals while revisiting label
normalization or prediction structure for open-vocabulary identities.

## Exact identities

- observation cohort specification:
  `sha256:af401d5d093dce0cb4f0e5a915d03b0a9c8696fdae204606b298c1dbda31eb51`
- observation cohort manifest:
  `sha256:c3f5434fc14126b77dbd83fe2dcd960d38db9e2ae76892934e7a57d3d3f2edb8`
- annotation semantic specification:
  `sha256:94675d7a9977b33e3c7d66baa3f06ae33d14222a358a1893285f3c5104a87a01`
- generated annotation manifest:
  `sha256:95e8b714b82e73590a0a70e71aa459765bec93c760b935933145ef8dfacc4952`
- generated baseline report:
  `sha256:d717b11ea9196e49ef6678818da2a307b7c505f37e526d8c718c1592ec968f58`

Independent verification re-read all 110 annotation observations, recomputed
canonical digests, validated exact annotation-source identities, and loaded all
230,284 examples. The ignored generated annotation dataset occupies
approximately 128 MiB in this build.

## Learned follow-up

The first shared score-span encoder was trained against these eight labels and
is recorded in
`../../experiments/whole_score/representation_v1/report.md`. It proves the
split-safe learned-model and checkpoint machinery but does not beat nearest
centroid overall; class imbalance produces majority-like heads on several
targets.

The label-balanced, validation-only successor is recorded in
`../../experiments/whole_score/representation_v2/report.md`. Its mean validation
macro-F1 rises from `0.16901250` to `0.18149075`, but remains below nearest
centroid at `0.20774838`; the exposed test split was not evaluated again.

Neither encoder is a critic. Trajectory review/preference schema v2 can now
retain one explicit criterion for every original/candidate judgment, and the
critic corpus audit enforces run-level held-out separation. The real readiness
audit remains at zero preferences, so no analytical label or agent decision is
being relabeled as reward.
