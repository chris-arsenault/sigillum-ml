# theme_gen/ — the kernel-constrained theme generator

Author a theme's pinned obligations as a **kernel** (the item-list DSL — `frame`/`pin`/`harm`/
`phrase`/`kernel`), and the engine produces a spread of *dissimilar* candidate realizations of the
free space. It produces **variety, never a quality ranking** — human audition selects.

Submodules:
- `kerneldsl.py` — the kernel DSL + dataclasses (`ThemeKernel`/`ThemeFrame`/the pin types).
- `engine.py` — the constrained generator (pin enforcement, backtracking, the candidate spread).
- `corpus.py` — builds a melody corpus; `model.py` / `model_factored.py` — Markov melody models.
- `features.py`, `density.py` — feature extraction + kernel density / free-space measures.
- `audition_specs.py`, `model_specs.py` — the audition + model registries.
- `report.py` — the candidate-spread report.

Run it via `python -m generation.tools.gen_themes <kernel-module-or-file>` (see the root `README.md`).
Architecture record: `docs/architecture/theme_generator.md`.
