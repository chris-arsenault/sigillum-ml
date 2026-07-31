# Label-balanced score-span encoder v2

This is a validation-only successor to the first shared representation pilot.
It tests the correction specified after v1—uniform label sampling inside each
target—without evaluating the already exposed test split again.

## Contract

The model architecture, Partitura-bound annotation manifest, eight targets, and
target-balanced optimization remain the same as v1. The only learning change is
that every target batch first samples labels uniformly and then samples an
example carrying each selected label. Checkpoint selection remains the
unweighted mean validation macro-F1 across all eight targets.

The run used 216 named features, 19,842 trainable parameters, seed `20260727`,
300 optimization steps, and validation every 50 steps. `evaluate_test` was
frozen to `false`; the generated report contains no v2 test metrics.

Frozen identities:

- annotation manifest:
  `sha256:95e8b714b82e73590a0a70e71aa459765bec93c760b935933145ef8dfacc4952`
- non-neural baseline report:
  `sha256:d717b11ea9196e49ef6678818da2a307b7c505f37e526d8c718c1592ec968f58`
- experiment specification:
  `sha256:4f4e5da9d9557fdd0f056b5fc33097cda83e47bac7a2b1400f3ad27fb834307e`

All three existing composition-lineage overlap checks passed.

## Development result

Step 100 was selected with mean validation macro-F1 `0.18149075`. That is
`0.01247825` above v1's validation score of `0.16901250`, but still below the
nearest-centroid mean validation macro-F1 of `0.20774838`.

| Target | Balanced v2 validation | Centroid validation | Delta |
| --- | ---: | ---: | ---: |
| Structural part relation | 0.076570 | 0.201775 | -0.125205 |
| Prominent part | 0.000000 | 0.013789 | -0.013789 |
| Material recurrence | 0.545013 | 0.586081 | -0.041068 |
| Form section | 0.068182 | 0.055026 | +0.013156 |
| Cadence type | 0.176746 | 0.178289 | -0.001543 |
| Harmonic function | 0.001303 | 0.000262 | +0.001041 |
| Orchestral role | 0.041578 | 0.101780 | -0.060202 |
| Seam boundary | 0.542534 | 0.524985 | +0.017549 |

Label balancing improves the mean development result and prevents several heads
from simply optimizing frequency, but it does not solve source-local open
vocabularies or beat the simple factual-feature baseline. This is model
development evidence, not a new generalization claim. A genuinely untouched
lineage cohort is still required before promoting a representation successor.

## Generated artifact identities

- model state:
  `sha256:fb30f476852745f47b7043ecf58b8d4c72149f449e81ad1abccddcb2f1d8b7e0`
- generated checkpoint file:
  `sha256:ea8a5dd650081bf2702f320bfee8b2b6d6eebddc5f34f832f8ae345b53866383`
- generated report:
  `sha256:b1212daf6cc7d644d3dc53891f8d7d0604f70c888bbf89d4a36b65e7dbaee2c8`

`python -m generation.tools.train_score_span_encoder --spec
experiments/whole_score/representation_v2/experiment.json verify`
independently reloaded and verified the checkpoint, experiment, and annotation
lineage.
