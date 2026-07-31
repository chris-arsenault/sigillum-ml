# Boundary-matched structural context v4

This experiment establishes a narrow learned signal for multimeasure
continuation. It does not train or claim a general musical-quality critic.

## Why v4 exists

Structural-context v1 learned to distinguish authentic four-measure successors
from random distant spans, but immediate boundary-profile distance was stronger:
`0.805986` versus `0.872435` validation score-macro accuracy.

V2 added the fixed boundary score as a zero-initialized residual base. Validation
selected step zero, proving that learned correction on the random-negative task
did not add value.

V3 changed the supervision rather than enlarging the network. For every anchor,
it selected a distant same-score negative whose first-measure boundary distance
most closely matched the authentic successor. That reduced the boundary
baseline to `0.654757`. The standalone hierarchy reached `0.637737`, narrowing
the gap to `-0.017020`.

V4 combines the boundary-matched task with the zero-initialized residual model.
The fixed local signal is always recoverable at step zero; the learned hierarchy
is selected only if longer context improves validation.

## Replicated validation result

All runs use the same architecture and hyperparameters. The seed controls model
initialization, minibatches, anchor sampling, and deterministic tie-breaking in
boundary matching. Test evaluation is disabled.

| Seed | Boundary baseline | Selected model | Delta | Selected step |
| ---: | ---: | ---: | ---: | ---: |
| 20260728 | 0.654757 | 0.699503 | +0.044746 | 50 |
| 20260729 | 0.691866 | 0.723408 | +0.031542 | 50 |
| 20260730 | 0.681450 | 0.716674 | +0.035225 | 50 |

Mean improvement is `+0.037171` with population standard deviation `0.005563`.
Every run selected step 50; continued optimization reduced validation accuracy.
This is stable development evidence that learned four-measure context adds
information beyond the controlled local boundary profile.

The primary frozen specification is
`sha256:234fcf9ef62c57b77da6692b7d3ddd7e63ff080a787976e3b63b7213a0bb2020`.
The primary checkpoint is
`sha256:f4d088c8fe07d09d0bc580f70fce87de03d097f39fc026602cc6f0a60dbd2571`.

## External holdout

Before inference, a separate PDMX cohort was frozen from composers absent from
the pilot training lineages. It contains seven scores, 2,548 measures, 100,251
events, and six Mahler, Schubert, Haydn, and Mendelssohn lineages. Exact source
digest and declared lineage overlap with the training manifest are both empty.

One initially considered Mahler 5 file was excluded before model evaluation
because Partitura rejected an invalid backup cursor before the start of a
measure. The parser was not weakened and the source was not repaired.

The primary checkpoint was evaluated once, without retraining or checkpoint
selection:

| Method | External score-macro accuracy |
| --- | ---: |
| Boundary-profile distance | 0.620833 |
| Selected v4 model | 0.717560 |
| Model minus baseline | +0.096726 |

The holdout has only six lineages, so this is positive external evidence rather
than a population-level performance claim. It is sufficient to retain the
representation as a structural-seam research signal.

Frozen holdout identities:

- cohort specification:
  `sha256:91753cb02d9d6f4020aa3cf52b3b7b970a1ca5381f4a160c0fd6abb9d522cbbd`
- generated observation manifest:
  `sha256:6d482187734486b18d61ec6660918754498b0313e54e64cd553e78ad66fcc9e4`

Generated checkpoints and machine-readable reports remain under ignored
`outputs/experiments/whole_score/structural_context_v4/`.

## Operational boundary

`StructuralSeamScorer` loads the selected checkpoint and scores consecutive
four-measure span pairs from a canonical `ScoreObservation`. It reports mean,
tenth-percentile, and minimum learned adjacency, the fixed boundary mean, the
mean learned residual, and the position of the worst successor boundary.

`StructuralSeamCritic` is the workflow adapter. Partitura attaches canonical
observations only for candidates that Ruby successfully compiled and exported;
the adapter emits the same threshold-free features in a learned critic result.
Those observations are request-scoped inputs and are not copied into persisted
trajectory assessments.

These are comparative features. There is no calibrated pass/fail threshold,
confidence value, candidate-selection policy, or claim that the signal measures
global quality. Ruby Partitura continues to own score parsing and candidate
execution; Python performs checkpoint inference only.
