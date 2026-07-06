"""Build the neural-theme training dataset from the curated, categorized corpus.

    python -m generation.tools.build_theme_dataset                 # full corpus, parallel -> outputs/datasets/theme_nn.jsonl
    python -m generation.tools.build_theme_dataset --limit 200     # a quick subset
    python -m generation.tools.build_theme_dataset --workers 1     # force serial

The per-file extraction (melody voice-scoring + harmony windows + figuration/motif + metric) is pure
CPU and independent per file, so it fans out across a process pool — the right parallelism here (no
model calls, so a Workflow's LLM agents would only add token cost without speeding it up).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from generation.theme_nn.dataset import DEFAULT_OUT, build_entry, save_jsonl, worklist


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else DEFAULT_OUT

    work = worklist()
    if args.limit:
        work = work[: args.limit]
    print(f"building {len(work)} examples on {args.workers} workers ...", flush=True)

    start = time.time()
    examples = []
    if args.workers <= 1:
        for i, task in enumerate(work, 1):
            examples.append(build_entry(task))
            if i % 500 == 0:
                print(f"  {i}/{len(work)}  ({time.time()-start:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(build_entry, work, chunksize=4), 1):
                examples.append(result)
                if i % 500 == 0:
                    print(f"  {i}/{len(work)}  ({time.time()-start:.0f}s)", flush=True)

    kept = [e for e in examples if e is not None]
    path = save_jsonl(kept, out)
    print(f"wrote {len(kept)}/{len(work)} examples to {path}  ({time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
