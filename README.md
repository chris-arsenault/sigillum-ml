# Sigillum ML

Theme generation, neural melody experiments, corpus ingestion, and related model
research split out of the main Sigillum symphony repo.

Expected sibling layout:

- `../sigillum-library` - Ruby Partitura score framework, analysis, and CLI
- `../sigillum` - current symphony materials used by some tests and examples

## Layout

- `generation/` - whole-score ML protocol and learned interfaces, theme
  generators, neural models, fractal/detailer code, and ML CLIs
- `experiments/` - model cards, registries, and frozen whole-score benchmarks
- `docs/architecture/` - theme-generator architecture notes
- `docs/research/` - neural and whole-score composition research
- `assets/raw/corpus/` - local corpus staging; ignored except for the top-level README
- `corpora/whole_score/` - tracked source registry for external whole-score
  material; fetched scores remain under the ignored raw corpus tree
- `tests/` - generation and model representation tests

## Tests

```bash
python -m pytest
```

`pyproject.toml` adds the sibling library and symphony repos to pytest's import path.
The Python tests execute without a retired compatibility framework: ML-specific
paths and token vocabularies live here, while score compilation and export cross
the Ruby Partitura CLI boundary. Missing legacy APIs are migration failures, not
a reason to skip their tests.

For whole-score composition, the boundary is stricter: Ruby Partitura owns
scheduling, candidate sandboxing, mechanical checks, promotion/rollback, and
trajectory/review/preference persistence plus blinded score rendering. Python
owns only learned proposers, critics, policies, features, weights, training,
versioned request/response DTOs, and read-only evidence/evaluation views.
