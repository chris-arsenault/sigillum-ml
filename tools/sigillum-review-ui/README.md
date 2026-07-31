# Sigillum Review UI

General local, user-run reviewer for Sigillum ML listening cadences.

The tool is deliberately cadence-agnostic. A generated manifest defines the
title, question, response options, items, variants, WAV files, and optional
artifact links. The UI serves WAV directly and atomically records one choice
per item in a gitignored JSON file.

Seam plausibility is the first manifest producer; future critic calibration,
candidate-selection, orchestration, form, and export-review cadences use the
same tool without changing the UI.

## Run

```bash
cd tools/sigillum-review-ui
npm install
npm run dev
```

The server defaults to Sulion's first published dev-server slot:
`0.0.0.0:26000`. It prints the user-facing URL when it starts. Use `PORT=26001`
through `PORT=26010` if the first slot is occupied.

The default inputs are:

- Manifest:
  `../../outputs/reviews/whole_score/seam_plausibility_v1/review-manifest.json`
- Media root:
  `../../outputs/reviews/whole_score/seam_plausibility_v1/`
- Results:
  `../../outputs/reviews/whole_score/seam_plausibility_v1/review-results.json`

Override them for another cadence:

```bash
SIGILLUM_REVIEW_MANIFEST=/path/to/review-manifest.json \
SIGILLUM_REVIEW_MEDIA_ROOT=/path/to/media-root \
SIGILLUM_REVIEW_RESULTS_FILE=/path/to/results.json \
PORT=26001 \
npm run dev
```

The server only exposes media paths referenced by the manifest, rejects path
traversal, cache-busts media by mtime and size, binds results to the manifest
digest, and writes results by temporary-file rename.

## Manifest v1

```json
{
  "schema_version": 1,
  "cadence_id": "example_v1",
  "title": "Example review",
  "description": "What this cadence tests.",
  "response": {
    "kind": "single_choice",
    "options": [
      { "id": "a", "label": "Prefer A" },
      { "id": "b", "label": "Prefer B" },
      { "id": "same", "label": "Same" }
    ]
  },
  "items": [
    {
      "id": "item-1",
      "title": "First comparison",
      "question": "Which is more coherent?",
      "tags": ["seam"],
      "variants": [
        {
          "id": "A",
          "label": "A",
          "audio": "bundles/item-1/A.wav",
          "links": [
            { "label": "MIDI", "path": "bundles/item-1/A.mid" }
          ]
        },
        {
          "id": "B",
          "label": "B",
          "audio": "bundles/item-1/B.wav",
          "links": []
        }
      ]
    }
  ]
}
```

## Verify

```bash
npm test
npm run typecheck
```
