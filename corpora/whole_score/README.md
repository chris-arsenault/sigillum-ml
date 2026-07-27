# Whole-score corpus registry

`sources.json` is the tracked source-of-truth for external whole-score training
material. The scores themselves are fetched into the ignored
`assets/raw/corpus/whole_score/` tree.

Two logical views are maintained:

- `research_full` admits musically useful material available for private,
  non-commercial research, while preserving its source and rights metadata.
- `redistributable_clean` contains only material with an explicit permissive
  score license or public-domain dedication. It is intended to make a future
  clean retraining run possible without reconstructing provenance.

List, fetch, and inspect sources with:

```bash
python -m generation.tools.fetch_composition_corpora list
python -m generation.tools.fetch_composition_corpora fetch openscore_orchestra
python -m generation.tools.fetch_composition_corpora fetch --view research_full
python -m generation.tools.fetch_composition_corpora status
```

Downloads are content-hashed and fetched atomically. Archives are extracted
with path traversal and link checks. A generated `fetch-state.json` records
successful downloads, failures, and exact local digests; `sources.json`
preserves source movement and availability notes.

Fetching a corpus does not make it a training example. Partitura Ruby must
validate and project score material into a versioned observation transport
before Python training code consumes it.

## Initial observation pilot

`pilot_v1.json` freezes the first whole-score representation-learning cohort:
94 OpenScore Orchestra scores and 16 S3 annotated symphony movements. The
selection is split by composition lineage, not by individual score file, so
alternate encodings and movements of one work cannot leak across train,
validation, and test.

Build and independently verify the ignored, content-addressed observations
with:

```bash
python -m generation.tools.build_score_observations discover
python -m generation.tools.build_score_observations build --jobs 2
python -m generation.tools.build_score_observations verify
```

The Python builder does not parse music. It invokes the sibling
`sigillum-library` Partitura CLI for each MusicXML/MXL source, validates the
canonical observation and source digests, attaches pinned annotation
provenance, and publishes a manifest only when every coverage gate passes.

The measured first build and the limitations of this cohort are recorded in
`pilot_v1_report.md`. Generated observations and their manifest stay under
ignored `outputs/`; the report and selection specification stay in Git.

## Annotation semantics pilot

`annotation_semantics_v1.json` binds the cohort's OpenScore Hauptstimme and S3
analytical annotations to canonical Partitura score addresses. It declares
eight representation targets and explicitly marks six candidate/critic targets
unavailable rather than treating analysis labels as musical-quality rewards.

Build, verify, and measure the ignored dataset with:

```bash
python -m generation.tools.build_annotation_semantics build --jobs 4
python -m generation.tools.build_annotation_semantics verify
python -m generation.tools.build_annotation_semantics baseline
```

Ruby Partitura owns annotation interpretation, score binding, factual feature
extraction, and source-quality warnings. Python schedules immutable
projections, enforces composition-lineage splits, validates digests, exposes
training views, and runs majority and nearest-centroid baselines. The measured
110-score result and its known source-alignment limitations are recorded in
`annotation_semantics_v1_report.md`.

## Retrieval status: 2026-07-27

All 15 registered fetchable sources downloaded, verified, and extracted. This
includes the historical POD and SOD archives: their old LOP project URLs have
not disappeared.

Two older aggregators remain unresolved and are intentionally not represented
as complete mirrors:

- The KernScores host refused the connection from this environment.
- The MuseData website remains online, but it no longer exposes one obvious
  full-database archive. Seven available, pinned repositories from the MuseData
  GitHub organization were fetched separately.

These are availability findings, not claims that the missing material no longer
exists anywhere. Recheck the registry homepages before treating either gap as
permanent.
