# Structural-context representation v1

This experiment tests one concrete prerequisite for recursive whole-score
composition: whether a learned hierarchical representation can recognize the
authentic continuation of a multimeasure score context.

## Frozen task

Each example contains a four-measure anchor, its authentic four-measure
successor, and a nonoverlapping span drawn at least twelve measures away from
the successor in the same score. The same-score negative prevents composer,
source, movement, and instrumentation identity from solving the task alone.
Composition lineages retain the train/validation/test assignments frozen in
the Partitura observation manifest.

Ruby Partitura remains the only score parser and semantic authority. Python
consumes its explicit measure, part, note, rest, unpitched-event, pitch, and
timing observations as ML inputs. The encoder learns a measure projection and
then a GRU representation over the ordered four-measure span. A directed
pairwise scorer ranks the authentic successor against the nonadjacent span.

The fixed comparison is `boundary_profile_distance`: squared distance between
the final anchor-measure profile and the first candidate-measure profile.
Chance pairwise accuracy is `0.5`.

Frozen identities:

- observation manifest:
  `sha256:c3f5434fc14126b77dbd83fe2dcd960d38db9e2ae76892934e7a57d3d3f2edb8`
- experiment specification:
  `sha256:597009c6b8659dbb598cd9467942ebffc1f403282d4be75789ca13c8a443f27d`

## Measured result

The dataset produced 2,872 train, 659 validation, and 678 test comparisons from
109 scores. One one-measure training score could not form an example and is
reported as omitted. The input has 142 factual dimensions per measure; the
learned model has 50,753 parameters. Validation selected step 350 of 400.

| Method | Validation score-macro accuracy | Test score-macro accuracy |
| --- | ---: | ---: |
| Boundary-profile distance | 0.872435 | 0.889790 |
| Hierarchical learned encoder | 0.805986 | 0.819469 |
| Learned minus baseline | -0.066449 | -0.070322 |
| Chance | 0.500000 | 0.500000 |

The controlled signal is learnable: the hierarchical model is substantially
above chance on held-out composition lineages. It does not, however, improve
on immediate boundary similarity. This is a negative representation result,
not a critic and not evidence that the model is ready to guide composition.
The result narrows the next model question: preserve the strong local boundary
signal and test whether learned longer-range context can add a residual
improvement, using validation-only development before evaluating a new
lineage holdout.

The existing test lineages were already exposed by earlier representation
work, so these test figures are diagnostic rather than an untouched
generalization claim.

Generated checkpoint and machine-readable report remain under ignored
`outputs/experiments/whole_score/structural_context_v1/`.
