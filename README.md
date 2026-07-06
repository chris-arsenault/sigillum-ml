# Sigillum ML

Theme generation, neural melody experiments, corpus ingestion, and related model
research split out of the main Sigillum symphony repo.

Expected sibling layout:

- `../sigillum-library` - shared framework and analysis code
- `../sigillum` - current symphony materials used by some tests and examples

## Layout

- `generation/` - theme generators, neural models, fractal/detailer code, and CLIs
- `experiments/` - model cards and experiment registries
- `docs/architecture/` - theme-generator architecture notes
- `docs/research/` - neural theme generation notes
- `assets/raw/corpus/` - local corpus staging; ignored except for the top-level README
- `tests/` - generation and model representation tests

## Tests

```bash
python -m pytest
```

`pyproject.toml` adds the sibling library and symphony repos to pytest's import path.
