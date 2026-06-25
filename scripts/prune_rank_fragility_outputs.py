#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


VERBOSE_FILENAMES = {
    "report.md",
    "audit_annotations.csv",
    "counterfactual_panel_manifest.csv",
    "counterfactual_scores.csv",
    "counterfactual_ranks.csv",
    "sota_margin_by_panel.csv",
    "kendall_tau_by_panel.csv",
    "per_example_advantage.csv",
    "Thumbs.db",
}

IMAGE_SUFFIXES = {".png", ".pdf"}


@dataclass(frozen=True)
class Target:
    path: Path
    category: str
    size: int
    is_dir: bool = False


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GiB"


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _validate_root(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.name != "rank_fragility":
        raise SystemExit(f"Refusing to prune outside a directory named 'rank_fragility': {resolved}")
    if not resolved.exists() or not resolved.is_dir():
        raise SystemExit(f"Prune root does not exist or is not a directory: {resolved}")
    return resolved


def collect_targets(root: Path) -> list[Target]:
    """Collect generated verbose rank-fragility artifacts eligible for pruning."""
    root = _validate_root(root)
    targets: dict[Path, Target] = {}

    for input_dir in root.rglob("_inputs"):
        if input_dir.is_dir():
            targets[input_dir] = Target(input_dir, "_inputs directories", _dir_size(input_dir), is_dir=True)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(parent.name == "_inputs" for parent in path.parents):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            targets[path] = Target(path, "plots", path.stat().st_size)
        elif path.name in VERBOSE_FILENAMES:
            targets[path] = Target(path, path.name, path.stat().st_size)

    return sorted(targets.values(), key=lambda target: str(target.path))


def summarize(targets: list[Target]) -> str:
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for target in targets:
        row = by_category[target.category]
        row[0] += 1
        row[1] += target.size

    lines = ["Prune target summary:"]
    total_count = 0
    total_size = 0
    for category in sorted(by_category):
        count, size = by_category[category]
        total_count += count
        total_size += size
        lines.append(f"  {category:34s} {count:5d}  {_format_size(size)}")
    lines.append(f"  {'TOTAL':34s} {total_count:5d}  {_format_size(total_size)}")
    return "\n".join(lines)


def prune(root: Path, apply: bool = False) -> list[Target]:
    targets = collect_targets(root)
    print(summarize(targets))
    if not apply:
        print("Dry run only. Re-run with --apply to delete these artifacts.")
        return targets

    for target in targets:
        if target.is_dir:
            shutil.rmtree(target.path)
        else:
            target.path.unlink()
    print(f"Deleted {len(targets)} verbose artifact(s).")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune verbose generated rank-fragility outputs.")
    parser.add_argument("--root", type=Path, default=Path("runs/rank_fragility"))
    parser.add_argument("--dry-run", action="store_true", help="Inspect targets without deleting them. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Delete allowlisted verbose artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.apply and args.dry_run:
        raise SystemExit("--apply and --dry-run are mutually exclusive")
    prune(args.root, apply=args.apply)


if __name__ == "__main__":
    main()
