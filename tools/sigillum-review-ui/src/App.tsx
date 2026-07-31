import { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

interface ReviewOption {
  id: string;
  label: string;
}

interface ReviewLink {
  label: string;
  url: string;
}

interface ReviewVariant {
  id: string;
  label: string;
  audioUrl: string;
  links: ReviewLink[];
}

interface ReviewItem {
  id: string;
  title: string;
  question: string;
  variants: ReviewVariant[];
  tags: string[];
}

interface ReviewPayload {
  schemaVersion: 1;
  cadenceId: string;
  title: string;
  description: string;
  response: {
    kind: "single_choice";
    options: ReviewOption[];
  };
  items: ReviewItem[];
  manifestDigest: string;
  resultsPath: string;
}

interface StoredResponse {
  itemId: string;
  choiceId: string;
  updatedAt: string;
}

interface ResultsPayload {
  schemaVersion: 1;
  cadenceId: string;
  manifestDigest: string;
  updatedAt: string;
  responses: Record<string, StoredResponse>;
}

type SaveState = "idle" | "saving" | "saved" | "error";

function App() {
  const [review, setReview] = useState<ReviewPayload | null>(null);
  const [responses, setResponses] = useState<Record<string, StoredResponse>>({});
  const [saveState, setSaveState] = useState<Record<string, SaveState>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [reviewPayload, resultPayload] = await Promise.all([
        fetchJson<ReviewPayload>("/api/review"),
        fetchJson<ResultsPayload>("/api/results")
      ]);
      if (
        reviewPayload.cadenceId !== resultPayload.cadenceId ||
        reviewPayload.manifestDigest !== resultPayload.manifestDigest
      ) {
        throw new Error("review results belong to another manifest");
      }
      setReview(reviewPayload);
      setResponses(resultPayload.responses);
      setLoadError(null);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const answered = Object.keys(responses).length;
  const complete = review ? answered === review.items.length : false;

  const choose = async (itemId: string, choiceId: string) => {
    setSaveState((state) => ({ ...state, [itemId]: "saving" }));
    try {
      const result = await fetchJson<ResultsPayload>(
        `/api/results/${encodeURIComponent(itemId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ choiceId })
        }
      );
      setResponses(result.responses);
      setSaveState((state) => ({ ...state, [itemId]: "saved" }));
    } catch {
      setSaveState((state) => ({ ...state, [itemId]: "error" }));
    }
  };

  if (loadError) {
    return <main className="load-error">{loadError}</main>;
  }
  if (!review) {
    return <main className="loading">Loading review</main>;
  }

  return (
    <div className="app-shell">
      <header className="review-header">
        <div>
          <p className="eyebrow">{review.cadenceId}</p>
          <h1>{review.title}</h1>
          {review.description && <p className="description">{review.description}</p>}
        </div>
        <div className={complete ? "progress complete" : "progress"}>
          <strong>
            {answered}/{review.items.length}
          </strong>
          <span>{complete ? "Complete" : "Reviewed"}</span>
        </div>
      </header>

      <main className="review-list">
        {review.items.map((item, index) => (
          <ReviewCard
            index={index}
            item={item}
            key={item.id}
            options={review.response.options}
            response={responses[item.id]}
            saveState={saveState[item.id] ?? "idle"}
            choose={choose}
          />
        ))}
      </main>

      <footer className={complete ? "review-footer complete" : "review-footer"}>
        <strong>{complete ? "Review complete" : "Choices save immediately"}</strong>
        <span>{review.resultsPath}</span>
      </footer>
    </div>
  );
}

function ReviewCard({
  index,
  item,
  options,
  response,
  saveState,
  choose
}: {
  index: number;
  item: ReviewItem;
  options: ReviewOption[];
  response: StoredResponse | undefined;
  saveState: SaveState;
  choose: (itemId: string, choiceId: string) => Promise<void>;
}) {
  const selectedChoice = response?.choiceId;
  const variantColumns = useMemo(
    () => ({ "--variant-count": String(item.variants.length) }) as React.CSSProperties,
    [item.variants.length]
  );

  return (
    <article className="review-card">
      <div className="item-heading">
        <div>
          <p className="item-number">Item {index + 1}</p>
          <h2>{item.title}</h2>
          <p>{item.question}</p>
        </div>
        <div className={`save-state ${saveState}`}>{saveLabel(saveState)}</div>
      </div>

      {item.tags.length > 0 && (
        <div className="tag-row">
          {item.tags.map((tag) => (
            <span className="tag" key={tag}>
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="variant-grid" style={variantColumns}>
        {item.variants.map((variant) => (
          <section className="variant" key={variant.id}>
            <h3>{variant.label}</h3>
            <audio controls preload="metadata" src={variant.audioUrl} />
            {variant.links.length > 0 && (
              <div className="variant-links">
                {variant.links.map((link) => (
                  <a href={link.url} key={link.label}>
                    {link.label}
                  </a>
                ))}
              </div>
            )}
          </section>
        ))}
      </div>

      <div className="choice-row" role="group" aria-label={item.question}>
        {options.map((option) => (
          <button
            className={selectedChoice === option.id ? "choice selected" : "choice"}
            disabled={saveState === "saving"}
            key={option.id}
            onClick={() => void choose(item.id, option.id)}
            type="button"
          >
            {option.label}
          </button>
        ))}
      </div>
    </article>
  );
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const value: unknown = await response.json();
  if (!response.ok) {
    const message =
      isErrorPayload(value)
        ? value.error
        : `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return value as T;
}

function isErrorPayload(value: unknown): value is { error: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "error" in value &&
    typeof value.error === "string"
  );
}

function saveLabel(state: SaveState): string {
  switch (state) {
    case "saving":
      return "Saving";
    case "saved":
      return "Saved";
    case "error":
      return "Save failed";
    default:
      return "";
  }
}

createRoot(document.getElementById("root")!).render(<App />);
