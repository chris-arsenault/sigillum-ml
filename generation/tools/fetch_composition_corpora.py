"""List, fetch, and inspect registered whole-score corpus sources."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from generation.composition.corpus_registry import (
    CorpusFetcher,
    CorpusRegistry,
    CorpusRegistryError,
)
from generation.project_paths import ROOT

DEFAULT_REGISTRY = ROOT / "corpora" / "whole_score" / "sources.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="source registry (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="list sources and logical views")
    listing.add_argument("--view", help="limit sources to one logical view")
    listing.add_argument("--json", action="store_true")

    fetch = commands.add_parser("fetch", help="fetch one or more source archives")
    fetch.add_argument("source_ids", nargs="*")
    fetch.add_argument("--view", help="fetch every source in one logical view")
    fetch.add_argument(
        "--no-extract",
        action="store_true",
        help="verify downloads without extracting archives",
    )

    status = commands.add_parser("status", help="show local source status")
    status.add_argument("--json", action="store_true")
    return parser


def _source_ids(
    registry: CorpusRegistry, source_ids: Sequence[str], view_id: str | None
) -> tuple[str, ...]:
    selected = list(source_ids)
    if view_id is not None:
        selected.extend(item.source_id for item in registry.view(view_id))
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise CorpusRegistryError("fetch requires source ids or --view")
    for source_id in selected:
        registry.source(source_id)
    return tuple(selected)


def _list(
    registry: CorpusRegistry, *, view_id: str | None, as_json: bool
) -> None:
    if view_id is None:
        sources = registry.sources
    else:
        sources = tuple(
            registry.source(selection.source_id)
            for selection in registry.view(view_id)
        )
    if as_json:
        print(
            json.dumps(
                {
                    "views": {
                        name: [
                            {
                                "source_id": selection.source_id,
                                "selection": selection.selection,
                            }
                            for selection in selections
                        ]
                        for name, selections in registry.views.items()
                    },
                    "sources": [
                        {
                            "id": source.id,
                            "title": source.title,
                            "roles": source.roles,
                            "rights": source.rights,
                            "artifacts": len(source.artifacts),
                        }
                        for source in sources
                    ],
                    "unresolved_sources": [
                        {
                            "id": source.id,
                            "title": source.title,
                            "status": source.status,
                            "note": source.note,
                        }
                        for source in registry.unresolved_sources
                    ],
                },
                indent=2,
            )
        )
        return
    for source in sources:
        tier = source.rights.get("tier", "unspecified")
        print(f"{source.id:34} {tier:32} {source.title}")
    if registry.unresolved_sources:
        print("\nUnresolved or moved sources:")
        for source in registry.unresolved_sources:
            print(f"{source.id:34} {source.status:32} {source.note}")


def _status(
    registry: CorpusRegistry, fetcher: CorpusFetcher, *, as_json: bool
) -> None:
    state = fetcher.status()
    if as_json:
        print(
            json.dumps(
                {
                    **state,
                    "unresolved_sources": [
                        {
                            "id": source.id,
                            "title": source.title,
                            "homepage": source.homepage,
                            "status": source.status,
                            "note": source.note,
                        }
                        for source in registry.unresolved_sources
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    source_state = state["sources"]
    for source in registry.sources:
        local = source_state.get(source.id, {})
        print(f"{source.id:34} {local.get('status', 'not_fetched')}")
        for artifact in source.artifacts:
            item = local.get("artifacts", {}).get(artifact.id, {})
            if item.get("status") == "blocked":
                print(f"  {artifact.id}: {item.get('error', 'unknown failure')}")
    for source in registry.unresolved_sources:
        print(f"{source.id:34} {source.status}: {source.note}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        registry = CorpusRegistry.load(arguments.registry)
        fetcher = CorpusFetcher(registry, project_root=ROOT, progress=print)
        if arguments.command == "list":
            _list(registry, view_id=arguments.view, as_json=arguments.json)
            return 0
        if arguments.command == "status":
            _status(registry, fetcher, as_json=arguments.json)
            return 0
        source_ids = _source_ids(
            registry, arguments.source_ids, arguments.view
        )
        results = fetcher.fetch(source_ids, extract=not arguments.no_extract)
        return 0 if all(results.values()) else 1
    except CorpusRegistryError as error:
        print(f"corpus error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
