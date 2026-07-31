import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

export interface ReviewOption {
  id: string;
  label: string;
}

export interface ReviewLink {
  label: string;
  path: string;
}

export interface ReviewVariant {
  id: string;
  label: string;
  audio: string;
  links: ReviewLink[];
}

export interface ReviewItem {
  id: string;
  title: string;
  question: string;
  variants: ReviewVariant[];
  tags: string[];
}

export interface ReviewManifest {
  schemaVersion: 1;
  cadenceId: string;
  title: string;
  description: string;
  response: {
    kind: "single_choice";
    options: ReviewOption[];
  };
  items: ReviewItem[];
}

export interface ReviewResponse {
  itemId: string;
  choiceId: string;
  updatedAt: string;
}

export interface ReviewResults {
  schemaVersion: 1;
  cadenceId: string;
  manifestDigest: string;
  updatedAt: string;
  responses: Record<string, ReviewResponse>;
}

export interface LoadedManifest {
  manifest: ReviewManifest;
  digest: string;
}

export async function loadManifest(manifestPath: string): Promise<LoadedManifest> {
  const source = await readFile(manifestPath, "utf8");
  return {
    manifest: parseManifest(JSON.parse(source) as unknown),
    digest: `sha256:${createHash("sha256").update(source).digest("hex")}`
  };
}

export function parseManifest(value: unknown): ReviewManifest {
  const root = object(value, "manifest");
  if (root.schema_version !== 1) {
    throw new Error("manifest schema_version must be 1");
  }
  const cadenceId = text(root.cadence_id, "cadence_id");
  const response = object(root.response, "response");
  if (response.kind !== "single_choice") {
    throw new Error("response.kind must be single_choice");
  }
  const options = array(response.options, "response.options").map((item, index) => {
    const option = object(item, `response.options[${index}]`);
    return {
      id: text(option.id, `response.options[${index}].id`),
      label: text(option.label, `response.options[${index}].label`)
    };
  });
  requireUnique(options.map((option) => option.id), "response option ids");
  if (options.length < 2) {
    throw new Error("response needs at least two options");
  }
  const items = array(root.items, "items").map((item, index) =>
    parseItem(item, index)
  );
  requireUnique(items.map((item) => item.id), "item ids");
  if (items.length === 0) {
    throw new Error("manifest needs at least one item");
  }
  return {
    schemaVersion: 1,
    cadenceId,
    title: text(root.title, "title"),
    description: optionalText(root.description),
    response: { kind: "single_choice", options },
    items
  };
}

export async function readResults(
  resultsPath: string,
  loaded: LoadedManifest
): Promise<ReviewResults> {
  try {
    const parsed = JSON.parse(await readFile(resultsPath, "utf8")) as unknown;
    return parseResults(parsed, loaded);
  } catch (error) {
    if (isMissingFile(error)) {
      return emptyResults(loaded);
    }
    throw error;
  }
}

export async function saveResponse(
  resultsPath: string,
  loaded: LoadedManifest,
  itemId: string,
  choiceId: string
): Promise<ReviewResults> {
  const item = loaded.manifest.items.find((candidate) => candidate.id === itemId);
  if (!item) {
    throw new Error(`unknown review item ${itemId}`);
  }
  const allowed = loaded.manifest.response.options.some(
    (option) => option.id === choiceId
  );
  if (!allowed) {
    throw new Error(`unknown review choice ${choiceId}`);
  }
  const results = await readResults(resultsPath, loaded);
  const updatedAt = new Date().toISOString();
  results.updatedAt = updatedAt;
  results.responses[itemId] = { itemId, choiceId, updatedAt };
  await writeResults(resultsPath, results);
  return results;
}

export function assertSafeRelativeMedia(relativePath: string): void {
  if (
    relativePath.length === 0 ||
    relativePath.includes("\0") ||
    path.isAbsolute(relativePath) ||
    relativePath.split(/[\\/]/).includes("..")
  ) {
    throw new Error("invalid media path");
  }
}

export function safeJoin(root: string, relativePath: string): string {
  assertSafeRelativeMedia(relativePath);
  const normalizedRoot = path.resolve(root);
  const resolved = path.resolve(normalizedRoot, relativePath);
  if (
    resolved !== normalizedRoot &&
    !resolved.startsWith(`${normalizedRoot}${path.sep}`)
  ) {
    throw new Error("invalid media path");
  }
  return resolved;
}

export function referencedMedia(manifest: ReviewManifest): Set<string> {
  const paths = new Set<string>();
  for (const item of manifest.items) {
    for (const variant of item.variants) {
      paths.add(variant.audio);
      for (const link of variant.links) {
        paths.add(link.path);
      }
    }
  }
  return paths;
}

function parseItem(value: unknown, index: number): ReviewItem {
  const item = object(value, `items[${index}]`);
  const variants = array(item.variants, `items[${index}].variants`).map(
    (rawVariant, variantIndex) => {
      const variant = object(
        rawVariant,
        `items[${index}].variants[${variantIndex}]`
      );
      const links = array(
        variant.links ?? [],
        `items[${index}].variants[${variantIndex}].links`
      ).map((rawLink, linkIndex) => {
        const link = object(
          rawLink,
          `items[${index}].variants[${variantIndex}].links[${linkIndex}]`
        );
        return {
          label: text(link.label, "review link label"),
          path: mediaPath(link.path, "review link path")
        };
      });
      return {
        id: text(variant.id, "variant id"),
        label: text(variant.label, "variant label"),
        audio: mediaPath(variant.audio, "variant audio"),
        links
      };
    }
  );
  requireUnique(variants.map((variant) => variant.id), `items[${index}] variant ids`);
  if (variants.length === 0) {
    throw new Error(`items[${index}] needs at least one variant`);
  }
  return {
    id: text(item.id, `items[${index}].id`),
    title: text(item.title, `items[${index}].title`),
    question: text(item.question, `items[${index}].question`),
    variants,
    tags: array(item.tags ?? [], `items[${index}].tags`).map((tag) =>
      text(tag, "tag")
    )
  };
}

function parseResults(value: unknown, loaded: LoadedManifest): ReviewResults {
  const root = object(value, "results");
  if (
    root.schema_version !== 1 ||
    root.cadence_id !== loaded.manifest.cadenceId ||
    root.manifest_digest !== loaded.digest
  ) {
    throw new Error("results do not belong to the current review manifest");
  }
  const responsesObject = object(root.responses ?? {}, "responses");
  const responses: Record<string, ReviewResponse> = {};
  for (const [itemId, rawResponse] of Object.entries(responsesObject)) {
    const response = object(rawResponse, `responses.${itemId}`);
    const parsedItemId = text(response.item_id, "response item_id");
    const choiceId = text(response.choice_id, "response choice_id");
    if (parsedItemId !== itemId) {
      throw new Error(`response key ${itemId} does not match item_id`);
    }
    const itemExists = loaded.manifest.items.some((item) => item.id === itemId);
    const choiceExists = loaded.manifest.response.options.some(
      (option) => option.id === choiceId
    );
    if (!itemExists || !choiceExists) {
      throw new Error(`results contain an unknown item or choice: ${itemId}`);
    }
    responses[itemId] = {
      itemId,
      choiceId,
      updatedAt: text(response.updated_at, "response updated_at")
    };
  }
  return {
    schemaVersion: 1,
    cadenceId: loaded.manifest.cadenceId,
    manifestDigest: loaded.digest,
    updatedAt: text(root.updated_at, "results updated_at"),
    responses
  };
}

function emptyResults(loaded: LoadedManifest): ReviewResults {
  return {
    schemaVersion: 1,
    cadenceId: loaded.manifest.cadenceId,
    manifestDigest: loaded.digest,
    updatedAt: new Date(0).toISOString(),
    responses: {}
  };
}

async function writeResults(
  resultsPath: string,
  results: ReviewResults
): Promise<void> {
  await mkdir(path.dirname(resultsPath), { recursive: true });
  const payload = {
    schema_version: results.schemaVersion,
    cadence_id: results.cadenceId,
    manifest_digest: results.manifestDigest,
    updated_at: results.updatedAt,
    responses: Object.fromEntries(
      Object.entries(results.responses).map(([itemId, response]) => [
        itemId,
        {
          item_id: response.itemId,
          choice_id: response.choiceId,
          updated_at: response.updatedAt
        }
      ])
    )
  };
  const temporary = `${resultsPath}.tmp`;
  await writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`);
  await rename(temporary, resultsPath);
}

function mediaPath(value: unknown, label: string): string {
  const parsed = text(value, label);
  assertSafeRelativeMedia(parsed);
  return parsed;
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function optionalText(value: unknown): string {
  return value === undefined ? "" : text(value, "description");
}

function requireUnique(values: string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`${label} must be unique`);
  }
}

function isMissingFile(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
