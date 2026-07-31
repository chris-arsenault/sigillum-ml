# Sigillum ML — Agent Guide

Sigillum ML provides learned proposals, critics, and policies around the Ruby
Partitura composition workflow. It is a Python research/tooling repo with a
Ruby sibling; it is **not** an Ahara-deployed cloud product.

## Read First

| Topic | Location |
| ---- | ---- |
| Current project state / handoff | [HANDOFF.md](HANDOFF.md) |
| Whole-score research direction | [docs/research/whole_score_composition_workflows.md](docs/research/whole_score_composition_workflows.md) |
| Theme generator architecture | [docs/architecture/theme_generator.md](docs/architecture/theme_generator.md) |
| Canonical score IR / workflow | `../sigillum-library` (Partitura) |
| Current symphony material | `../sigillum` |
| Platform conventions (only if a component is ever deployed) | `../ahara/INTEGRATION.md`, `../ahara-standards/standards/` |

## Architectural Boundary (do not blur)

- **Ruby Partitura owns musical semantics:** score parsing, composition graph,
  scheduling, candidate sandbox execution, mechanical validation,
  MusicXML/MIDI export, promotion, rollback, and trajectory persistence. Live
  in `../sigillum-library`.
- **Python owns ML only:** datasets, tensorization of canonical Partitura
  observations, learned features/weights, training, inference, and read-only
  protocol DTOs.
- Do not create a second Python score runtime and do not move music-generation
  semantics out of the Ruby DSL. Reuse `../sigillum-library`; do not copy its
  code here.

## Composition Content Rules

- For newly composed score sources, write the note lists themselves. Do not
  generate sounding material with helpers, loops, comprehensions, repeaters,
  transposers, pattern expanders, or any code that stamps out notes.
- Use `../sigillum` only for current symphony material required by tests or an
  explicit experiment. Prefer real score excerpts over invented studies when an
  experiment claims to test "real" material.

## Verify Before Handoff

There is no `make ci` here; the canonical checks are explicit:

```bash
# Python (run from repo root)
python -m pytest -ra -o addopts=          # expect zero skips

# Ruby Partitura sibling, when a change touches the workflow it exercises
cd ../sigillum-library && bundle exec ruby -Ipartitura/test \
  -e 'Dir["partitura/test/test_*.rb"].sort.each { |f| require_relative f }'

git diff --check                          # in every repo you touched
```

- Run the relevant suite before considering work done for anything beyond a
  small doc edit. Prefer behavioral/contract tests over source-text or
  fake-absence tests.
- Add or update tests when you change pure helpers, data transforms, protocol
  DTOs, or a user-visible workflow.

## Generated Artifacts Stay Out of Git

- Model outputs, corpora, MLflow runs, checkpoints, rendered MusicXML/MIDI/WAV,
  review media, and per-run results JSON are generated and gitignored
  (`outputs/**`, `models/**`, `*.pt`, etc.). Never commit them.
- Durable, checked-in artifacts are: source (including hand-written score
  material), experiment contracts, tools, tests, and measured reports.

## Human Review / Audition Tooling

Model direction is validated by human listening. Follow the established peer
pattern (Lindelion `tools/lamath-review-ui/`), implemented here as
`tools/sigillum-review-ui/` — do **not** invent a delivery
mechanism per task and do **not** stand up an unmanaged background server:

- The review UI is the **checked-in, local, dev-only**
  `tools/sigillum-review-ui/`; new cadences add manifests, not new apps.
- It **reads a manifest** produced by the real Partitura workflow plus
  regeneratable, gitignored media, and **writes a gitignored
  results/preference JSON**. The tool code is durable; its media and results are
  not.
- It is run **on demand by the user** (for example, `npm run dev`), never
  launched by an agent in the background. Start dev servers only when the user
  explicitly asks.
- Bind to the published Sulion dev-server slots only: `0.0.0.0`, ports
  `26000`-`26010`. The server must print its user-facing URL rather than
  assuming the container's own address. Do not use arbitrary ports; they may
  not be published to the user.
- "Ready for review" means the checked-in tool can be started by the documented
  command and is reachable from the user's browser at its printed Sulion URL —
  not a `localhost`/`127.0.0.1` process, not an agent-held server, and not files
  under ignored `outputs/`.
- Do not build an Ahara-deployed (Cognito/RDS/ALB) service for a local review;
  that is the wrong boundary for this repo unless the user explicitly asks for
  a deployed product.

## Git & Secrets

- Work on `main` by default; do not create or switch branches unless asked.
- Commit only what the task asked for; preserve unrelated working-tree changes.
- Push only when the user explicitly asks in the current turn; a prior push
  request is not standing permission.
- Never commit secrets or `.env` files. Use `with-cred -- <command>` on the
  first attempt for any secret-backed command; treat a credential failure as a
  hard stop, not a puzzle to route around.

## Working Style

- Prefer complete, well-factored features over throwaway slices; if a temporary
  slice is necessary, make the boundary explicit.
- Distinguish local checks from deployed/hosted behavior; state exactly what was
  and was not verified.
- Keep research claims honest: report negative results, do not present
  experimenter-authored labels as human preference, and retain the boundary
  baseline as a mandatory comparison for any learned signal.
