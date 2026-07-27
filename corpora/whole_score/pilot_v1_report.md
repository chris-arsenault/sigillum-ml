# Whole-score observation pilot v1

This report records the first complete build of
`whole_score_observation_pilot_v1`. It is a durable measurement of the pinned
cohort in `pilot_v1.json`, not a copy of the generated dataset.

## Purpose and boundary

The pilot establishes the score-to-representation boundary needed before model
training:

- Ruby Partitura is the sole authority for MusicXML/MXL interpretation.
- Python discovers pinned sources, schedules projections, validates canonical
  digests, joins annotation provenance, and exposes read-only splits.
- Generated observations, manifests, corpora, and future model artifacts remain
  outside Git.
- A successful projection is not a positive reward label and does not imply
  compositional quality.

The representation is intended to support structural relationships, material
identity, phrase and section context, seam context, and candidate-to-original
change. Future critics remain criterion-specific: coherence, identity, seams,
orchestration, and reserve. Mechanical validity and requirement binding remain
Ruby-authoritative auxiliary signals.

## Pinned cohort

| Collection | Version | Scores |
| --- | --- | ---: |
| OpenScore Orchestra | v1.0.1, commit `45edea23b6fcb9e64a8d56b0d86b9cdfb99614f7` | 94 |
| S3 Symbolic Symphonies | commit `f845a46314fc603259bbe9bdf773a7bb2d235295` | 16 |
| **Total** | | **110** |

The 24 composition lineages are assigned as units:

| Split | Scores |
| --- | ---: |
| Train | 79 |
| Validation | 16 |
| Test | 15 |

OpenScore prominent-line MusicXML files are annotation artifacts rather than
independent score targets. S3 annotation CSV and text files are likewise joined
as provenance; they are not interpreted as critic rewards in this slice.

## Measured build

The complete build finished on 2026-07-27 with every coverage gate passing:

| Measurement | Count |
| --- | ---: |
| Scores projected | 110 |
| Failed scores | 0 |
| Partitura warnings | 0 |
| Parts | 1,965 |
| Measures | 32,247 |
| Timed events | 1,793,062 |
| Pitched notes | 1,240,984 |
| Annotation files | 1,122 |
| Annotation rows | 350,573 |
| Scores with annotations | 110 |

The ignored generated dataset occupies approximately 59 MiB in this build.
Independent verification re-read all 110 observations, recomputed their
canonical digests, checked their source hashes, and verified the manifest.

Exact identities:

- observation schema: `1`
- projector: `partitura-score-observation-v1`
- cohort specification:
  `sha256:af401d5d093dce0cb4f0e5a915d03b0a9c8696fdae204606b298c1dbda31eb51`
- generated manifest:
  `sha256:c3f5434fc14126b77dbd83fe2dcd960d38db9e2ae76892934e7a57d3d3f2edb8`

The manifest identity is local generated state. Rebuilding from the pinned
sources must reproduce the observation identities and coverage, while the
manifest itself also records its generation timestamp.

## What is ready

This cohort is ready for deterministic and learned representation experiments.
Dataset consumers can select train, validation, or test without rediscovering
files or reinterpreting score syntax. The versioned annotation projection and
first non-neural measurements are recorded in
`annotation_semantics_v1_report.md`.

## What is not ready

This is not yet a critic-training or reinforcement-learning dataset:

- there are no candidate/original edit pairs in this corpus;
- there are no blind human preferences attached to these scores;
- deterministic or agent-generated trajectories must not be treated as
  positive examples merely because they were generated;
- the test split is a score-corpus holdout and is separate from the
  `held_out_evaluation` exclusion used for human preferences.

The annotation projection supplies analytical representation targets, not
critic rewards. Learned critics should follow only after criterion-specific
candidate comparisons and human labels exist.
