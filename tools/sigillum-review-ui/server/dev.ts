import express from "express";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer as createViteServer } from "vite";

import {
  loadManifest,
  readResults,
  referencedMedia,
  safeJoin,
  saveResponse,
  type LoadedManifest,
  type ReviewManifest
} from "./review.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(appRoot, "../..");

const manifestPath = envPath(
  "SIGILLUM_REVIEW_MANIFEST",
  "outputs/reviews/whole_score/seam_plausibility_v1/review-manifest.json"
);
const mediaRoot = path.resolve(
  process.env.SIGILLUM_REVIEW_MEDIA_ROOT ?? path.dirname(manifestPath)
);
const resultsPath = envPath(
  "SIGILLUM_REVIEW_RESULTS_FILE",
  "outputs/reviews/whole_score/seam_plausibility_v1/review-results.json"
);
const host = process.env.HOST ?? "0.0.0.0";
const port = Number(process.env.PORT ?? 26000);
const publicHost = process.env.SULION_PUBLIC_HOST ?? "192.168.66.3";

interface PublicLink {
  label: string;
  url: string;
}

interface PublicVariant {
  id: string;
  label: string;
  audioUrl: string;
  links: PublicLink[];
}

interface PublicItem {
  id: string;
  title: string;
  question: string;
  variants: PublicVariant[];
  tags: string[];
}

interface PublicReview {
  schemaVersion: 1;
  cadenceId: string;
  title: string;
  description: string;
  response: ReviewManifest["response"];
  items: PublicItem[];
  manifestDigest: string;
  resultsPath: string;
}

async function main() {
  const loaded = await loadManifest(manifestPath);
  const allowedMedia = referencedMedia(loaded.manifest);
  const app = express();
  app.use(express.json({ limit: "32kb" }));

  app.get("/api/review", async (_request, response, next) => {
    try {
      response.json(await publicReview(loaded));
    } catch (error) {
      next(error);
    }
  });

  app.get("/api/results", async (_request, response, next) => {
    try {
      response.json(await readResults(resultsPath, loaded));
    } catch (error) {
      next(error);
    }
  });

  app.put("/api/results/:itemId", async (request, response, next) => {
    try {
      const choiceId = request.body?.choiceId;
      if (typeof choiceId !== "string") {
        response.status(400).json({ error: "choiceId must be a string" });
        return;
      }
      response.json(
        await saveResponse(resultsPath, loaded, request.params.itemId, choiceId)
      );
    } catch (error) {
      next(error);
    }
  });

  app.get("/media/*", async (request, response, next) => {
    try {
      const relativePath = decodeMediaPath(
        (request.params as { 0?: string })[0] ?? ""
      );
      if (!allowedMedia.has(relativePath)) {
        response.status(404).end();
        return;
      }
      const filePath = safeJoin(mediaRoot, relativePath);
      await stat(filePath);
      response.type(mediaType(relativePath));
      createReadStream(filePath).pipe(response);
    } catch (error) {
      next(error);
    }
  });

  const vite = await createViteServer({
    root: appRoot,
    server: { middlewareMode: true },
    appType: "spa"
  });
  app.use(vite.middlewares);

  app.use(
    (
      error: unknown,
      _request: express.Request,
      response: express.Response,
      _next: express.NextFunction
    ) => {
      const message = error instanceof Error ? error.message : String(error);
      response.status(500).json({ error: message });
    }
  );

  app.listen(port, host, () => {
    console.log(`Sigillum review UI: http://${publicHost}:${port}`);
    console.log(`Listening: http://${host}:${port}`);
    console.log(`Cadence: ${loaded.manifest.cadenceId}`);
    console.log(`Manifest: ${manifestPath}`);
    console.log(`Results: ${resultsPath}`);
  });
}

function envPath(name: string, fallback: string): string {
  return path.resolve(repoRoot, process.env[name] ?? fallback);
}

async function publicReview(loaded: LoadedManifest): Promise<PublicReview> {
  return {
    schemaVersion: 1,
    cadenceId: loaded.manifest.cadenceId,
    title: loaded.manifest.title,
    description: loaded.manifest.description,
    response: loaded.manifest.response,
    items: await Promise.all(
      loaded.manifest.items.map(async (item) => ({
        id: item.id,
        title: item.title,
        question: item.question,
        tags: item.tags,
        variants: await Promise.all(
          item.variants.map(async (variant) => ({
            id: variant.id,
            label: variant.label,
            audioUrl: await mediaUrl(variant.audio),
            links: await Promise.all(
              variant.links.map(async (link) => ({
                label: link.label,
                url: await mediaUrl(link.path)
              }))
            )
          }))
        )
      }))
    ),
    manifestDigest: loaded.digest,
    resultsPath
  };
}

async function mediaUrl(relativePath: string): Promise<string> {
  const stats = await stat(safeJoin(mediaRoot, relativePath));
  const encoded = relativePath
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `/media/${encoded}?v=${Math.round(stats.mtimeMs)}-${stats.size}`;
}

function decodeMediaPath(value: string): string {
  return value
    .split("/")
    .map((segment) => decodeURIComponent(segment))
    .join("/");
}

function mediaType(relativePath: string): string {
  switch (path.extname(relativePath).toLowerCase()) {
    case ".wav":
      return "audio/wav";
    case ".mid":
    case ".midi":
      return "audio/midi";
    case ".musicxml":
    case ".xml":
      return "application/xml";
    default:
      return "application/octet-stream";
  }
}

void main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
