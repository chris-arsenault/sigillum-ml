# Sigillum ML handoff

Date: 2026-07-29  
Working repository: `/home/sulion/repos/sigillum-ml`  
Sibling score authority: `/home/sulion/repos/sigillum-library`

## Read this first

The project has substantial uncommitted Phase 6 work and one narrow positive
machine-learning result. It does **not** yet have a human-validated critic,
generative whole-score model, adaptive controller, or RL result.

The latest agent repeatedly overvalued infrastructure, verification, and
self-defined slices. It then claimed that an ignored, agent-local HTML package
was ready for user review and served it from `127.0.0.1`. That was not a valid
delivery: the user could not access an agent-local server, the reviewed media
was not durable, and the package did not contain real composition-workflow
candidates. Do not repeat this.

The temporary HTTP server has been stopped. The user has not completed a
review and no human preference evidence was collected.

## Repository state

Neither repository's Phase 6 work is committed or pushed.

- `sigillum-ml`
  - branch: `main`
  - HEAD and `origin/main`: `2d5ee444ae1c655bf2286d5bc94a7b5d535794c1`
  - many modified and untracked Phase 6 files; use `git status --short`
- `sigillum-library`
  - branch: `main`
  - HEAD and `origin/main`: `d4445ac5afe7b9b74864fb6a8366f2a9ad3a35bf`
  - many modified Partitura workflow/protocol files; use `git status --short`

Preserve both working trees. Do not stage indiscriminately, delete code to
declare a phase unsuccessful, or infer that the published base contains the
work described below.

The active Sulion plan is
`9811b1ed-002c-4cef-a7f2-84104b58cb66`. Phases 1-5 are complete, Phase 6 is
`in_progress`, and Phases 7-9 are pending. Keep Phase 6 active.

## Architectural boundary

Keep this boundary sharp:

- Ruby Partitura owns score parsing, musical semantics, composition graphs,
  scheduling, source patches, sandbox execution, mechanical validation,
  MusicXML/MIDI export, promotion, rollback, and trajectory persistence.
- Python owns ML datasets, tensorization of canonical Partitura observations,
  learned features and weights, training, inference, and read-only protocol
  DTOs.
- Do not create a second Python score runtime or move music-generation
  semantics out of the Ruby DSL.
- `SelectionRequest.candidate_observations` is request-scoped. It carries
  canonical observations only for candidates Ruby successfully exported and is
  deliberately excluded from persisted trajectory assessments.

## Research evidence

### Analytical representation learning: negative

Partitura projected 110 scores into 230,284 lineage-separated factual examples
covering eight analytical targets.

- representation V1:
  - learned test macro-F1: `0.17711850`
  - nearest-centroid test macro-F1: `0.21505413`
  - result: learned encoder did not beat the factual-feature baseline
- label-balanced V2:
  - validation macro-F1: `0.18149075`
  - nearest-centroid validation macro-F1: `0.20774838`
  - result: improvement over V1, still negative; test was not reopened

These labels are score facts, not composition rewards.

### Structural-context V4: positive but narrow

V1 learned authentic four-measure continuation but lost to immediate boundary
distance. V2 retained the boundary score as a residual but validation selected
step zero. V3 made negatives boundary-matched. V4 combined boundary-matched
negatives with a zero-initialized residual model.

V4 validation-only replications:

| Seed | Boundary baseline | V4 | Delta |
| ---: | ---: | ---: | ---: |
| 20260728 | 0.654757 | 0.699503 | +0.044746 |
| 20260729 | 0.691866 | 0.723408 | +0.031542 |
| 20260730 | 0.681450 | 0.716674 | +0.035225 |

Mean improvement: `+0.037171`.

Frozen external PDMX holdout:

- seven scores, six unseen lineages
- boundary score-macro accuracy: `0.620833`
- V4 score-macro accuracy: `0.717560`
- delta: `+0.096726`

This supports only the claim that learned four-measure context adds information
beyond immediate boundary similarity on the defined continuation task.

It does **not** establish musical quality, a threshold, confidence, pass/fail
authority, candidate selection, a general critic, or a composition policy.

Primary durable report:
`experiments/whole_score/structural_context_v4/report.md`.

### Critic learning: no result

The five-head pairwise critic implementation and its corpus/readiness audit
exist. The actual preference corpus count is zero and
`outputs/datasets/whole_score/critic_v1/` is absent. No critic checkpoint has
been trained.

Do not relabel historical annotations or medium-quality agent decisions as
expert rewards. Agent work may generate candidate alternatives, but it must not
supply the ground-truth preference.

### Controller and RL: not started

There is no adaptive-controller experiment and no RL pilot. Do not start RL
from the uncalibrated seam signal.

## Failed review delivery

The following checked-in experiment/tooling was added:

- `experiments/whole_score/seam_review_v1/`
- `generation/tools/build_seam_review.py`
- `tests/test_seam_review.py`

Generated ignored output exists at:

`outputs/reviews/whole_score/seam_review_v1/`

It contains:

- six blinded comparisons, one per external lineage;
- twelve WAV files;
- a public static page;
- a private answer key with authentic/model/baseline identities.

The package verifies successfully, but it is **not an acceptable completed
deliverable**:

1. It is under ignored `outputs/`, so the media is not durable across clones.
2. The HTML was exposed through an agent-local server rather than a
   user-accessible artifact.
3. It compares historical authentic continuations with boundary-matched
   synthetic splices, not real alternatives emitted by the composition
   workflow.
4. The audio is an additive rendering of canonical Partitura note events, not
   the normal Partitura MusicXML/MIDI audition path.
5. The six cases were selected as a targeted model-versus-baseline diagnostic,
   not an unbiased accuracy sample.

Do not cite this package as human evaluation evidence.

## Existing-source blockers discovered during the review attempt

Four existing Movement IV Ruby DSL variants were considered as real candidate
material. All currently fail the stricter Partitura compiler:

- `mvt4_CLAUDE_dsl.rb`: `bar_onsets_cross_barline` in
  `s1_clarinet_dbl_bass_clarinet`
- `mvt4_dance_0709_CODEX_dsl.rb`: `bar_onsets_cross_barline` in
  `s2_alto_flute_dbl_flute`
- `mvt4_storyteller_merged.rb`: `bar_onsets_cross_barline` in `m2_flute`
- `mvt4_storyteller_CODEX_dsl.rb`: `bar_onsets_cross_barline` in
  `s1_violoncello`

The corresponding legacy Python assemblers also fail immediately with
`ModuleNotFoundError: No module named 'framework'`. Do not restore Python music
framework code to bypass this. Either choose currently valid Partitura material
or repair only the concrete Ruby score sources needed for an actual experiment.

## What the next agent should accomplish

The next unit of work is a **real, durable, user-accessible musical review
artifact**, not another framework.

Completion requires all of the following:

1. Use mechanically valid alternatives produced through the real Ruby
   composition workflow, not historical span splices.
2. Render the normal Partitura MusicXML/MIDI audition artifacts.
3. Present a very small review batch requiring only A/B/Same decisions and no
   command line, server startup, file assembly, taxonomy choice, or written
   explanation from the user.
4. Deliver it through a user-visible durable artifact/document mechanism.
   `localhost`, `127.0.0.1`, an agent process, or an unexplained path under
   ignored outputs does not count.
5. Verify access from the user's side of the interface before claiming it is
   ready.
6. After the choices arrive, analyze them immediately against the frozen V4 and
   boundary scores and write the result durably, including negative findings.

If no supported durable media-delivery mechanism is available, stop and report
that exact access boundary before building more review infrastructure.

Do not create a narrower goal and mark it complete merely because a generator,
page, test suite, or local process exists.

## Suggested `AGENTS.md` guardrail

The current instructions were already sufficient; this is not an excuse for
the prior execution failure. If the user wants a durable additional guardrail,
the useful addition is:

> “Ready for review” means directly accessible from the user's interface
> without starting a server, running a command, or navigating generated
> directories. Localhost URLs and agent-local processes never count. The
> reviewed content, not merely its generator, must be durable and portable.
> Autonomous research continues through a substantive domain result and
> user-visible artifact; do not narrow the goal merely to reach a terminal
> state.

Do not edit `AGENTS.md` unless the user explicitly asks.

## Generated local data present on this machine

These are ignored and must remain outside Git:

| Path | Current size |
| --- | ---: |
| `outputs/datasets/whole_score/annotation_semantics_v1/` | 1.1 GiB |
| `outputs/experiments/whole_score/representation_v1/` | 132 KiB |
| `outputs/experiments/whole_score/representation_v2/` | 128 KiB |
| `outputs/experiments/whole_score/structural_context_v4/` | 676 KiB |
| `outputs/datasets/whole_score/structural_context_external_holdout_v1/` | 42 MiB |
| `outputs/reviews/whole_score/seam_review_v1/` | 16 MiB |

Primary V4 checkpoint:

`outputs/experiments/whole_score/structural_context_v4/checkpoint.pt`

Checkpoint digest:

`sha256:f4d088c8fe07d09d0bc580f70fce87de03d097f39fc026602cc6f0a60dbd2571`

## Last verification

Current final verification:

- Python: `117 passed`, zero skips
- Partitura: `243 runs / 1,464 assertions`, zero skips
- DSL integration: `4 runs / 693 assertions`, zero skips
- review-package verification:
  - six items
  - six lineages
  - twelve non-silent stereo WAV files
  - exact shared anchor audio within every pair
  - no private identity/score leakage into the public page
- `git diff --check`: passed in both repositories

Useful commands:

```bash
cd /home/sulion/repos/sigillum-ml
git status --short
python -m pytest -ra -o addopts=
python -m generation.tools.build_seam_review verify

cd /home/sulion/repos/sigillum-library
git status --short
bundle exec ruby -Ipartitura/test \
  -e 'Dir["partitura/test/test_*.rb"].sort.each { |file| require_relative file }'
bundle exec ruby -Ipartitura/test \
  partitura/test/integration/test_technique_library_dsl.rb
```
