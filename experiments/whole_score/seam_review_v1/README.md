# Structural seam human review v1

This is a six-item usability-sized review of the selected structural-context
V4 checkpoint. Each item presents the same four-measure anchor followed by one
of two continuations from the same unseen historical score:

- the authentic following four measures;
- the boundary-matched distant span used by the frozen external evaluation.

The selection takes one item per external-holdout lineage. It preferentially
chooses cases where the learned model ranks the authentic continuation above
the splice while the fixed boundary baseline does not. This is therefore a
targeted disagreement review, not an unbiased estimate of population accuracy.

The public review page contains no composer, score, source, authentic-side,
model-score, or baseline-score information. The private answer key remains in
the ignored generated output. Audio is rendered only from canonical Partitura
observations at a fixed tempo; Python does not parse MusicXML.

Build and verify:

```bash
python -m generation.tools.build_seam_review build
python -m generation.tools.build_seam_review verify
```

Generated audio, the public page, and the private answer key live under
`outputs/reviews/whole_score/seam_review_v1/` and remain outside Git.
