#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_BENCHMARK_FILES = [
    "utils/loader.py",
    "utils/baselines.py",
    "utils/__init__.py",
]
DEFAULT_ANALYSIS_FILE = "utils/analysis.py"
IGNORED_EXTERNAL_PREFIXES = (
    "self.",
    "logging.",
    "logger.",
)
CLASS_SUMMARY_OVERRIDES = {
    "ResultWriter": "Persists analysis and benchmarking artifacts to output directories.",
    "BaselineParams": "Dataclass container for baseline-model hyperparameters and fingerprint settings.",
    "BaseLoader": "Abstract loader interface for producing standardized benchmark splits.",
    "TDCLoader": "Benchmark loader for Therapeutics Data Commons datasets via API clients.",
    "TabularLoader": "Flexible file-based benchmark loader with column inference and split normalization.",
    "PolarisLoader": "Benchmark loader that adapts Polaris train/test splits into BenchAudit format.",
    "DTILoader": "DTI-specific extension of `TabularLoader` that enforces sequence requirements.",
    "AnalyzerConfig": "Schema for configuring molecular and DTI audit parameters.",
    "AnalysisResult": "Structured return object containing summaries and row-level audit outputs.",
    "StretcherAlignment": "Typed representation of global sequence-alignment statistics.",
    "PSAStretcherAligner": "Cached adapter around `psa.stretcher` for repeated pairwise alignments.",
    "SMILESAnalyzer": "Core molecular audit engine for contamination, similarity, conflicts, and cliffs.",
    "DTIAnalyzer": "Extended audit engine that combines molecular and target-level diagnostics.",
}
METHOD_INTENT_OVERRIDES = {
    "__init__": "Constructs and initializes class state.",
    "get_splits": "Builds and returns standardized benchmark data splits.",
    "_maybe_clean": "Applies optional SMILES cleaning and normalizes canonical columns.",
    "_import_from_str": "Imports a class or callable from a dotted Python path.",
    "_init_dataset": "Initializes dataset backends and resolves the requested benchmark source.",
    "_pick": "Selects the first matching column from a list of candidate names.",
    "_read_like": "Reads tabular data from CSV, TSV, or Parquet based on file extension.",
    "_resolve_column": "Resolves a configured or inferred column name from candidate aliases.",
    "_standardize_cols": "Renames and normalizes semantic columns required by downstream analysis.",
    "_prepare_split": "Normalizes a split dataframe and enforces valid sequence columns for DTI.",
    "_build_drug_summary": "Computes drug-level overlap and hygiene summary statistics.",
    "_analyze_sequences": "Computes sequence leakage, redundancy, and alignment-based diagnostics.",
    "_analyze_structures_foldseek": "Computes structure-level leakage diagnostics using Foldseek alignments.",
    "_featurize_for_similarity": "Computes molecular and scaffold fingerprints for similarity calculations.",
    "_normalize_seq": "Normalizes amino-acid sequence strings for stable alignment behavior.",
    "_invert_alignment": "Reorients a sequence alignment object by swapping query and subject fields.",
    "_empty_alignment": "Builds a placeholder alignment result for missing or invalid sequence pairs.",
    "run": "Executes the class's primary workflow and returns computed outputs.",
    "write_summary": "Writes summary statistics to `summary.json`.",
    "write_performance": "Writes baseline benchmarking metrics to `performance.json`.",
    "_write_jsonl": "Writes drill-down records to line-delimited JSON files.",
    "write_records": "Writes per-record audit tables to `records.csv`.",
    "write_analysis": "Persists all generated analysis artifacts and returns file paths.",
    "align": "Computes a pairwise sequence alignment with caching and normalization.",
}


@dataclass(frozen=True)
class DataclassFieldDoc:
    name: str
    annotation: str
    default: Optional[str]


@dataclass(frozen=True)
class MethodDoc:
    name: str
    signature: str
    lineno: int
    end_lineno: int
    loc: int
    visibility: str
    docstring: Optional[str]
    call_targets: List[str]
    internal_calls: List[str]
    external_calls: List[str]
    assertion_counts: List[str]


@dataclass(frozen=True)
class ClassDoc:
    category: str
    name: str
    source_path: str
    lineno: int
    end_lineno: int
    bases: List[str]
    decorators: List[str]
    docstring: Optional[str]
    dataclass_fields: List[DataclassFieldDoc]
    methods: List[MethodDoc]


def _safe_unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node).strip()
    except Exception:
        return "<unparseable>"


def _clean_docstring(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return None
    cleaned = doc.strip()
    return cleaned or None


def _first_sentence(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return None
    first = doc.split("\n\n", 1)[0].strip()
    return " ".join(first.split()) if first else None


def _title_from_identifier(identifier: str) -> str:
    text = identifier.strip("_")
    if not text:
        return identifier
    if text.startswith("test_"):
        text = text[5:]
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").strip()
    if not text:
        return identifier
    return text[0].upper() + text[1:]


def _method_intent(name: str, docstring: Optional[str]) -> str:
    if name in METHOD_INTENT_OVERRIDES:
        return METHOD_INTENT_OVERRIDES[name]

    explicit = _first_sentence(docstring)
    if explicit:
        return explicit

    if name == "__init__":
        return "Constructs and initializes class state."

    normalized = name.strip("_")
    words = _title_from_identifier(normalized).rstrip(".")
    parts = words.split()
    if not parts:
        return "Implements class-specific behavior."

    first = parts[0].lower()
    tail = " ".join(parts[1:])
    verb_map = {
        "get": "Retrieves",
        "set": "Sets",
        "build": "Builds",
        "run": "Executes",
        "write": "Writes",
        "read": "Reads",
        "load": "Loads",
        "save": "Saves",
        "analyze": "Analyzes",
        "prepare": "Prepares",
        "normalize": "Normalizes",
        "resolve": "Resolves",
        "standardize": "Standardizes",
        "import": "Imports",
        "pick": "Selects",
        "align": "Computes",
        "featurize": "Computes features for",
    }
    if first in verb_map:
        phrase = verb_map[first]
        if tail:
            return f"{phrase} {tail.lower()}."
        return f"{phrase} class-specific outputs."

    return f"Performs {words.lower()}."


def _ordered_unique(items: Iterable[str], *, limit: Optional[int] = None) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def _collect_call_name(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _collect_call_name(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    if isinstance(expr, ast.Call):
        return _collect_call_name(expr.func)
    if isinstance(expr, ast.Subscript):
        return _collect_call_name(expr.value)
    return None


def _method_visibility(name: str) -> str:
    if name.startswith("__") and name.endswith("__"):
        return "dunder"
    if name.startswith("_"):
        return "private"
    return "public"


def _is_builtin_call(target: str) -> bool:
    root = target.split(".", 1)[0]
    return hasattr(builtins, root)


def _collect_method_docs(node: ast.ClassDef) -> List[MethodDoc]:
    methods: List[MethodDoc] = []
    for stmt in node.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        signature = f"{stmt.name}({_safe_unparse(stmt.args)})"
        if stmt.returns is not None:
            signature = f"{signature} -> {_safe_unparse(stmt.returns)}"

        calls = []
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call):
                name = _collect_call_name(child.func)
                if name:
                    calls.append(name)

        internal_calls = _ordered_unique(
            target for target in calls if target.startswith("self.") and not target.startswith("self.assert")
        )
        external_calls = _ordered_unique(
            target
            for target in calls
            if not target.startswith(IGNORED_EXTERNAL_PREFIXES)
            and not _is_builtin_call(target)
            and not target.startswith("self.assert")
        )

        assertion_counts: List[str] = []
        assertions = [target.split(".", 1)[1] for target in calls if target.startswith("self.assert")]
        if assertions:
            counts = Counter(assertions)
            assertion_counts = [f"{name} ({count})" for name, count in sorted(counts.items())]

        loc = 1
        if getattr(stmt, "end_lineno", None) is not None:
            loc = max(1, int(stmt.end_lineno) - int(stmt.lineno) + 1)

        methods.append(
            MethodDoc(
                name=stmt.name,
                signature=signature,
                lineno=int(stmt.lineno),
                end_lineno=int(getattr(stmt, "end_lineno", stmt.lineno)),
                loc=loc,
                visibility=_method_visibility(stmt.name),
                docstring=_clean_docstring(ast.get_docstring(stmt)),
                call_targets=_ordered_unique(calls),
                internal_calls=internal_calls,
                external_calls=external_calls,
                assertion_counts=assertion_counts,
            )
        )
    return methods


def _collect_dataclass_fields(node: ast.ClassDef) -> List[DataclassFieldDoc]:
    fields: List[DataclassFieldDoc] = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(
                DataclassFieldDoc(
                    name=stmt.target.id,
                    annotation=_safe_unparse(stmt.annotation),
                    default=_safe_unparse(stmt.value) if stmt.value is not None else None,
                )
            )
        elif (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            fields.append(
                DataclassFieldDoc(
                    name=stmt.targets[0].id,
                    annotation="Any",
                    default=_safe_unparse(stmt.value),
                )
            )
    return fields


def _collect_class_docs(paths: Sequence[Path], category: str, root: Path) -> List[ClassDoc]:
    class_docs: List[ClassDoc] = []
    for path in sorted(paths, key=lambda p: p.as_posix()):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel_path = path.relative_to(root).as_posix()
        for stmt in module.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            class_docs.append(
                ClassDoc(
                    category=category,
                    name=stmt.name,
                    source_path=rel_path,
                    lineno=int(stmt.lineno),
                    end_lineno=int(getattr(stmt, "end_lineno", stmt.lineno)),
                    bases=[_safe_unparse(base) for base in stmt.bases],
                    decorators=[_safe_unparse(dec) for dec in stmt.decorator_list],
                    docstring=_clean_docstring(ast.get_docstring(stmt)),
                    dataclass_fields=_collect_dataclass_fields(stmt),
                    methods=_collect_method_docs(stmt),
                )
            )
    return class_docs


def _render_class_doc(doc: ClassDoc) -> List[str]:
    lines: List[str] = []
    lines.append(f"### `{doc.name}`")
    lines.append(f"- Source: `{doc.source_path}:{doc.lineno}`")
    lines.append(f"- Category: `{doc.category}`")
    lines.append(f"- Base classes: {', '.join(f'`{base}`' for base in doc.bases) if doc.bases else 'None'}")
    lines.append(
        f"- Decorators: {', '.join(f'`{decorator}`' for decorator in doc.decorators) if doc.decorators else 'None'}"
    )
    lines.append(f"- Class size: {doc.end_lineno - doc.lineno + 1} lines")
    lines.append(
        f"- Summary: {CLASS_SUMMARY_OVERRIDES.get(doc.name) or _first_sentence(doc.docstring) or _title_from_identifier(doc.name)}"
    )

    if doc.docstring:
        lines.append("- Docstring:")
        lines.append("```text")
        lines.extend(doc.docstring.splitlines())
        lines.append("```")

    if doc.dataclass_fields:
        lines.append("- Data fields:")
        for field in doc.dataclass_fields:
            default = f" = `{field.default}`" if field.default is not None else ""
            lines.append(f"  - `{field.name}: {field.annotation}`{default}")

    if not doc.methods:
        lines.append("- Methods: none")
        lines.append("")
        return lines

    lines.append(f"- Methods ({len(doc.methods)} total):")
    for method in doc.methods:
        lines.append(f"  - `{method.name}`")
        lines.append(f"    - Signature: `{method.signature}`")
        lines.append(f"    - Source: `{doc.source_path}:{method.lineno}`")
        lines.append(f"    - Visibility: `{method.visibility}`")
        lines.append(f"    - Method size: {method.loc} lines")
        lines.append(f"    - Intent: {_method_intent(method.name, method.docstring)}")
        if method.assertion_counts:
            lines.append(f"    - Assertions: {', '.join(f'`{a}`' for a in method.assertion_counts)}")
        if method.internal_calls:
            internal = ", ".join(f"`{name}`" for name in method.internal_calls[:12])
            lines.append(f"    - Internal calls: {internal}")
        if method.external_calls:
            external = ", ".join(f"`{name}`" for name in method.external_calls[:12])
            lines.append(f"    - External calls: {external}")
    lines.append("")
    return lines


def render_markdown(benchmark_docs: Sequence[ClassDoc], analysis_docs: Sequence[ClassDoc]) -> str:
    lines: List[str] = []
    lines.append("# Benchmarking and Analysis Class Reference")
    lines.append("")
    lines.append(
        "This file is auto-generated by `scripts/generate_benchmark_analysis_class_docs.py`. "
        "Do not edit manually."
    )
    lines.append("")
    lines.append("## Repository Purpose")
    lines.append(
        "BenchAudit is a reproducible audit framework for molecular property and "
        "drug-target interaction (DTI) benchmarks. It standardizes molecular representations, "
        "quantifies split hygiene and similarity, identifies label inconsistencies and activity cliffs, "
        "and optionally trains baseline predictive models for sanity checking."
    )
    lines.append("")
    lines.append("## Pipeline Structure")
    lines.append("1. Configuration ingestion: `run.py` reads one or more YAML benchmark configs.")
    lines.append(
        "2. Data loading: `utils/loader.py` normalizes split data into canonical "
        "tables (`train`, `valid`, `test`) with consistent column names."
    )
    lines.append(
        "3. Data auditing: `utils/analysis.py` computes contamination, similarity, "
        "label conflicts, and (for DTI) sequence/structure alignment diagnostics."
    )
    lines.append(
        "4. Optional baseline benchmarking: `utils/baselines.py` trains models on train "
        "split features and evaluates on test split labels."
    )
    lines.append(
        "5. Artifact persistence: `utils/__init__.py` (`ResultWriter`) serializes "
        "summary and drill-down outputs to disk."
    )
    lines.append("")
    lines.append("## Expected Inputs")
    lines.append("| Input | Format | Scientific Role |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| Benchmark configuration | YAML (`configs/*.yml`) | Declares benchmark type, task "
        "(`classification` or `regression`), split source paths/API metadata, and preprocessing parameters. |"
    )
    lines.append(
        "| Molecular records | CSV/TSV/Parquet or API-provided tables | Provides ligands (SMILES) and labels "
        "for train/valid/test auditing and model evaluation. |"
    )
    lines.append(
        "| DTI target information (optional) | Sequence and target-ID columns | Enables cross-split "
        "target leakage checks and sequence-level redundancy/alignment analysis. |"
    )
    lines.append(
        "| Foldseek metadata/alignment files (optional) | JSONL and `.m8` | Enables structure-aware "
        "target similarity diagnostics when available. |"
    )
    lines.append("")
    lines.append("## Produced Outputs")
    lines.append("| Output Artifact | Format | Interpretation |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| `summary.json` | JSON | Compact benchmark-level statistics: split sizes, contamination, "
        "similarity summaries, conflict/cliff counts, and DTI-specific diagnostics. |"
    )
    lines.append(
        "| `records.csv` | CSV | Row-level canonical audit table with cleaned structures, labels, split tags, "
        "and optional target annotations. |"
    )
    lines.append(
        "| `conflicts.jsonl` | JSONL | Detailed records of inconsistent labels among duplicate or comparable entities. |"
    )
    lines.append(
        "| `cliffs.jsonl` | JSONL | Detailed activity-cliff pairs with high similarity but discordant labels. |"
    )
    lines.append(
        "| `sequence_alignments.jsonl` | JSONL | DTI sequence alignment diagnostics for cross-split target overlap. |"
    )
    lines.append(
        "| `structure_alignments.jsonl` | JSONL | DTI structure-level alignment diagnostics when Foldseek inputs exist. |"
    )
    lines.append(
        "| `performance.json` | JSON | Baseline model predictions and evaluation metrics (when benchmarking is enabled). |"
    )
    lines.append("")
    lines.append("## Operational Interface")
    lines.append(
        "- Primary CLI: `python run.py --config <path.yml> --out-root runs [--benchmark]`"
    )
    lines.append(
        "- Batch mode: `python run.py --configs configs --out-root runs [--benchmark]`"
    )
    lines.append(
        "- Scientific interpretation: the audit path (`--benchmark` omitted) quantifies dataset hygiene; "
        "the benchmark path (`--benchmark`) adds model-based performance context."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Benchmarking classes: {len(benchmark_docs)}")
    lines.append(f"- Analysis classes: {len(analysis_docs)}")
    lines.append("")

    lines.append("## Benchmarking Classes")
    lines.append("")
    if benchmark_docs:
        for doc in benchmark_docs:
            lines.extend(_render_class_doc(doc))
    else:
        lines.append("No classes discovered.")
        lines.append("")

    lines.append("## Analysis Classes")
    lines.append("")
    if analysis_docs:
        for doc in analysis_docs:
            lines.extend(_render_class_doc(doc))
    else:
        lines.append("No classes discovered.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _resolve_relative_paths(root: Path, rel_paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for rel in rel_paths:
        path = (root / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing source file: {rel}")
        out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate markdown reference docs for benchmarking classes and analysis classes."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark_and_analysis_class_reference.md"),
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--benchmark-files",
        nargs="*",
        default=DEFAULT_BENCHMARK_FILES,
        help="Relative source files that define benchmarking classes.",
    )
    parser.add_argument(
        "--analysis-file",
        default=DEFAULT_ANALYSIS_FILE,
        help="Relative source file that defines analysis classes.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether output file is up to date instead of writing it.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    benchmark_paths = _resolve_relative_paths(root, args.benchmark_files)
    analysis_paths = _resolve_relative_paths(root, [args.analysis_file])

    benchmark_docs = _collect_class_docs(benchmark_paths, category="benchmarking", root=root)
    analysis_docs = _collect_class_docs(analysis_paths, category="analysis", root=root)

    markdown = render_markdown(benchmark_docs, analysis_docs)
    output_path = (root / args.output).resolve()

    if args.check:
        current = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if current != markdown:
            print(f"[docs] {output_path.relative_to(root)} is out of date.")
            print("[docs] Run scripts/generate_benchmark_analysis_class_docs.py and commit the result.")
            return 1
        print(f"[docs] {output_path.relative_to(root)} is up to date.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[docs] wrote {output_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
