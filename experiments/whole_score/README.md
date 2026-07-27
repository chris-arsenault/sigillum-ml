# Whole-score evaluation lab

This directory contains frozen benchmark definitions, not generated outputs or
claimed results. A benchmark manifest pins each brief and starting Partitura
score by SHA-256, declares all baseline strategies and required ablations, and
defines the complete run and blinded-comparison matrix.

Generated score sources, trajectories, run JSONL, private review mappings,
preferences, MusicXML, MIDI, and reports belong under ignored experiment
storage such as `outputs/evaluation/`; they must not be committed here.

The smoke manifest proves the contract and tooling with one deliberately small
case. It is not a musically meaningful benchmark result. Production studies
must add multiple frozen briefs, multiple seeds, independent raters, and power
analysis before making comparative claims.

Commands:

```bash
python -m generation.tools.evaluate_composition verify \
  experiments/whole_score/v1_smoke/manifest.json

python -m generation.tools.evaluate_composition collect \
  experiments/whole_score/v1_smoke/manifest.json \
  --runs outputs/evaluation/v1_smoke/runs.jsonl \
  --source PATH_TO_COMPLETED_SCORE.rb \
  --case kernel-study --strategy deterministic_graph \
  --seed 1729 --trajectory PATH_TO_TRAJECTORY.jsonl \
  --model-calls 3 --wall-seconds 12.5

python -m generation.tools.evaluate_composition report \
  experiments/whole_score/v1_smoke/manifest.json \
  --runs outputs/evaluation/v1_smoke/runs.jsonl \
  --reviews outputs/evaluation/v1_smoke/reviews.jsonl \
  --preferences outputs/evaluation/v1_smoke/preferences.jsonl
```

Partitura owns score measurement and blinded score rendering. This Python lab
only verifies frozen experiment inputs, stores ML run metadata, joins held-out
human evidence, and aggregates plural diagnostics without producing a
composite reward or automatic winner.
