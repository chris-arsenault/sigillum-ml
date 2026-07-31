import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  parseManifest,
  referencedMedia,
  safeJoin,
  saveResponse
} from "./review.js";

const manifestValue = {
  schema_version: 1,
  cadence_id: "fixture",
  title: "Fixture review",
  description: "A generic cadence.",
  response: {
    kind: "single_choice",
    options: [
      { id: "a", label: "Prefer A" },
      { id: "b", label: "Prefer B" },
      { id: "same", label: "Same" }
    ]
  },
  items: [
    {
      id: "item-1",
      title: "First item",
      question: "Which is stronger?",
      tags: ["fixture"],
      variants: [
        {
          id: "A",
          label: "A",
          audio: "bundles/one/A.wav",
          links: [{ label: "MIDI", path: "bundles/one/A.mid" }]
        },
        {
          id: "B",
          label: "B",
          audio: "bundles/one/B.wav",
          links: []
        }
      ]
    }
  ]
};

test("generic manifest preserves cadence-defined choices and media", () => {
  const manifest = parseManifest(manifestValue);
  assert.equal(manifest.cadenceId, "fixture");
  assert.deepEqual(
    manifest.response.options.map((option) => option.id),
    ["a", "b", "same"]
  );
  assert.deepEqual(
    [...referencedMedia(manifest)].sort(),
    ["bundles/one/A.mid", "bundles/one/A.wav", "bundles/one/B.wav"]
  );
});

test("manifest rejects paths outside the configured media root", () => {
  const unsafe = structuredClone(manifestValue);
  unsafe.items[0].variants[0].audio = "../private.wav";
  assert.throws(() => parseManifest(unsafe), /invalid media path/);
  assert.throws(() => safeJoin("/tmp/media", "../private.wav"), /invalid media path/);
});

test("response persistence is atomic and bound to the manifest digest", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "sigillum-review-"));
  try {
    const manifest = parseManifest(manifestValue);
    const loaded = { manifest, digest: `sha256:${"1".repeat(64)}` };
    const resultsPath = path.join(directory, "results.json");
    const results = await saveResponse(resultsPath, loaded, "item-1", "a");
    assert.equal(results.responses["item-1"].choiceId, "a");
    const stored = JSON.parse(await readFile(resultsPath, "utf8")) as {
      manifest_digest: string;
      responses: Record<string, { choice_id: string }>;
    };
    assert.equal(stored.manifest_digest, loaded.digest);
    assert.equal(stored.responses["item-1"].choice_id, "a");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
