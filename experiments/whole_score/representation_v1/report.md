# Shared score-span encoder pilot v1

This report records the first frozen learned-representation experiment over the
Partitura-bound analytical dataset. The checked-in `experiment.json` is the
experiment contract. The generated checkpoint and machine-readable report
remain under ignored `outputs/`.

## Contract

The pilot tests whether named factual score-span features can be transformed
into one learned representation shared across eight analytical targets:

- each scalar value is paired with a learned embedding of its Partitura feature
  name;
- a shared token encoder and mean pool produce a 32-dimensional score-span
  representation;
- each target has its own classification head and train-only label vocabulary;
- feature normalization is fit only on training examples;
- each optimization step samples the same number of examples from every target;
- checkpoint selection uses the unweighted mean of the eight validation
  macro-F1 values;
- the test split is evaluated once after restoring the selected checkpoint.

The model has 216 feature names and 19,842 trainable parameters. The run used
seed `20260727`, 300 optimization steps, 32 examples per target per step, and
validation every 50 steps. It ran on CPU in approximately 68 seconds.

Target balancing does not imply label balancing within a target. This
distinction matters to the result below.

## Frozen inputs

- annotation manifest:
  `sha256:95e8b714b82e73590a0a70e71aa459765bec93c760b935933145ef8dfacc4952`
- non-neural baseline report:
  `sha256:d717b11ea9196e49ef6678818da2a307b7c505f37e526d8c718c1592ec968f58`
- experiment specification:
  `sha256:c34b5e000c33dec645688cdbbcc453d1d5cf83046e2911add8b0d28633311190`

All three composition-lineage overlap checks passed with empty overlap.

## Selection and held-out result

Step 250 was selected with mean validation macro-F1 `0.16901250`. The restored
checkpoint achieved mean test macro-F1 `0.17711850`.

For context, the mean test macro-F1 across the same targets is `0.16583200` for
the majority baseline and `0.21505413` for nearest centroid. The learned model
therefore beats majority by `0.01128650` but trails nearest centroid by
`0.03793563`.

| Target | Validation macro-F1 | Learned test macro-F1 | Centroid test macro-F1 | Delta |
| --- | ---: | ---: | ---: | ---: |
| Structural part relation | 0.079741 | 0.089157 | 0.190021 | -0.100864 |
| Prominent part | 0.014610 | 0.007871 | 0.013078 | -0.005207 |
| Material recurrence | 0.438332 | 0.447901 | 0.626852 | -0.178951 |
| Form section | 0.057143 | 0.007702 | 0.003205 | +0.004497 |
| Cadence type | 0.243257 | 0.344358 | 0.246101 | +0.098257 |
| Harmonic function | 0.004139 | 0.003114 | 0.002712 | +0.000402 |
| Orchestral role | 0.087335 | 0.091621 | 0.119862 | -0.028241 |
| Seam boundary | 0.427543 | 0.425224 | 0.518602 | -0.093378 |

The learned model improves materially on cadence and very slightly on form and
harmony, but it collapses to majority-like behavior for relations, recurrence,
and seams. It does not displace the nearest-centroid baseline.

## Interpretation

The implementation proves the operational machinery: learned feature
embeddings and encoder weights are trained jointly, target heads share one
representation, split lineage is enforced, checkpoint lineage is
content-addressed, selection is validation-only, and artifacts are independently
verifiable.

The representation result itself is negative. Equal target sampling prevents
the relation corpus from dominating the smaller tasks, but ordinary
cross-entropy still follows the majority labels inside each head. The validation
history already shows this collapse: several targets remain exactly at their
majority macro-F1 across checkpoints.

A successor should be specified from training statistics and validation
behavior with label-balanced sampling or class-balanced loss, plus structured
labels for open-vocabulary instrument, form, and harmony identities. Because
this run has now exposed the current test split, it must not be repeatedly used
to tune that successor and then described as untouched. A fresh lineage-level
holdout is required before making a new generalization claim.

This encoder is analytical representation learning, not a musical critic. The
historical annotations still contain no candidate/original preferences or
criterion-specific quality judgments.

## Generated artifact identities

- model state:
  `sha256:c0b67262e2aa10e18669d2c7316e30df7584f2a3f9139fd169827a613ea92566`
- generated checkpoint file:
  `sha256:31f5c5e0ac34542592d697cfea13e392ee447d645c3c2261ffe52f00b5f71895`
- generated report:
  `sha256:d3f446f29c259a35d38995aa87ef80cc186d222bfb2c04f3d14656a449863f93`

`python -m generation.tools.train_score_span_encoder verify` independently
reloaded the checkpoint and report and verified their model, experiment, and
annotation-manifest lineage.
