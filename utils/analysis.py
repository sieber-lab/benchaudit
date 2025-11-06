from __future__ import annotations
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple, Literal
import logging
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import Chem
try:
    from Levenshtein import ratio as _levenshtein  # type: ignore
except ImportError:  # pragma: no cover - fallback when python-Levenshtein is absent
    from difflib import SequenceMatcher

    def _levenshtein(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
from rdkit.Chem.Scaffolds.MurckoScaffold import (
    MakeScaffoldGeneric as _GraphFramework,
    GetScaffoldForMol as _GetScaffoldForMol,
)

try:
    import psa
except ImportError:  # pragma: no cover - optional dependency
    psa = None


def _to_python_scalar(val: Any) -> Any:
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return [_to_python_scalar(v) for v in list(val)]
    if isinstance(val, (np.generic,)):
        return val.item()
    if isinstance(val, (pd.Timestamp,)):
        return val.to_pydatetime()
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    return val


def _is_sequence_label(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _normalize_label_list(value: Any) -> List[Any]:
    if isinstance(value, pd.Series):
        value = value.tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        seq = list(value)
    else:
        seq = [value]
    out: List[Any] = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray, pd.Series)):
            out.extend(_normalize_label_list(item))
            continue
        if item is None:
            out.append(None)
            continue
        if isinstance(item, (np.generic,)):
            item = item.item()
        if isinstance(item, str):
            s = item.strip()
            if s == "" or s.lower() in {"nan", "na", "null"}:
                out.append(None)
                continue
            try:
                fv = float(s)
                if fv.is_integer():
                    out.append(int(fv))
                else:
                    out.append(fv)
                continue
            except ValueError:
                out.append(s)
                continue
        try:
            if pd.isna(item):
                out.append(None)
                continue
        except TypeError:
            pass
        out.append(item)
    return out


def _label_to_tuple(value: Any) -> Tuple[Any, ...]:
    return tuple(_normalize_label_list(value))


def _series_has_sequence_labels(series: pd.Series) -> bool:
    if series.empty:
        return False
    return series.apply(_is_sequence_label).any()


def _label_series_to_matrix(series: pd.Series) -> np.ndarray:
    if series.empty:
        return np.zeros((0, 0), dtype=float)
    rows = [_normalize_label_list(v) for v in series.tolist()]
    max_len = max((len(row) for row in rows), default=0)
    if max_len == 0:
        return np.zeros((len(rows), 0), dtype=float)
    arr = np.full((len(rows), max_len), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if val is None:
                arr[i, j] = np.nan
            else:
                try:
                    arr[i, j] = float(val)
                except (TypeError, ValueError):
                    arr[i, j] = np.nan
    return arr


def _classification_delta(a: Tuple[Any, ...], b: Tuple[Any, ...]) -> List[Optional[float]]:
    length = max(len(a), len(b))
    out: List[Optional[float]] = []
    for idx in range(length):
        va = a[idx] if idx < len(a) else None
        vb = b[idx] if idx < len(b) else None
        if va is None or vb is None:
            out.append(None)
            continue
        try:
            out.append(int(va) - int(vb))
        except (TypeError, ValueError):
            try:
                out.append(float(va) - float(vb))
            except (TypeError, ValueError):
                out.append(None)
    return out


def _regression_delta(a: Any, b: Any) -> List[Optional[float]]:
    seq_a = _normalize_label_list(a)
    seq_b = _normalize_label_list(b)
    length = max(len(seq_a), len(seq_b))
    out: List[Optional[float]] = []
    for idx in range(length):
        va = seq_a[idx] if idx < len(seq_a) else None
        vb = seq_b[idx] if idx < len(seq_b) else None
        if va is None or vb is None:
            out.append(None)
            continue
        try:
            out.append(float(va) - float(vb))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _delta_exceeds_threshold(delta: List[Optional[float]], sigma) -> bool:
    if not delta:
        return False
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigmas = list(sigma)
    else:
        sigmas = [sigma] * len(delta)
    if not sigmas:
        return False
    for idx, d in enumerate(delta):
        if d is None:
            continue
        thr = sigmas[idx] if idx < len(sigmas) else sigmas[-1]
        if thr is None or thr == 0:
            continue
        try:
            if abs(float(d)) >= float(thr):
                return True
        except (TypeError, ValueError):
            continue
    return False


@dataclass
class AnalyzerConfig:
    """Minimal, YAML-friendly config for SMILES analysis.

    Parameters
    ----------
    task_type : Literal['classification', 'regression']
        Whether labels are classes or real-valued.
    typ : Literal['tdc', 'tabular', 'polaris']
        Type of the task.
    sim_threshold : float
        Consensus similarity threshold (based on MoleculeACE): a pair is 'similar'
        if >= threshold for at least one of:
          - molecular ECFP Tanimoto
          - scaffold (generic Murcko) ECFP Tanimoto
          - normalized SMILES Levenshtein similarity
    fp_radius : int
        ECFP/Morgan radius.
    fp_nbits : int
        ECFP/Morgan bit length.
    smiles_col : Optional[str]
        If input dataframes do NOT already have 'smiles_clean', rename this column to 'smiles_clean'.
    label_col : Optional[str]
        If input dataframes do NOT already have 'label_raw', rename this column to 'label_raw'.
    id_col : Optional[str]
        If provided and present, use/rename as 'id'. Otherwise sequential ids will be assigned per split.
    label_cols : Optional[List[str]]
        Optional list of multi-task label column names. When present, labels are treated as lists.
    sequence_col : Optional[str]
        Optional column name for amino-acid sequences (renamed to 'sequence_aa' when present).
    target_id_col : Optional[str]
        Optional column name for target identifiers (renamed to 'target_id' when present).
    """

    task_type: Literal["classification", "regression"]
    typ: Literal["tdc", "tabular", "polaris"]
    sim_threshold: float = 0.9
    fp_radius: int = 2
    fp_nbits: int = 2048
    smiles_col: Optional[str] = None
    label_col: Optional[str] = None
    id_col: Optional[str] = None
    label_cols: Optional[List[str]] = None
    sequence_col: Optional[str] = None
    target_id_col: Optional[str] = None


@dataclass
class AnalysisResult:
    summary: Dict[str, Any]
    per_record_df: pd.DataFrame
    conflicts_rows: List[Dict[str, Any]]
    cliffs_rows: List[Dict[str, Any]]
    sequence_alignment_rows: Optional[List[Dict[str, Any]]] = None


def _normalize_columns(df: pd.DataFrame, cfg: AnalyzerConfig, split: str) -> pd.DataFrame:
    """Ensure expected columns exist: "smiles_clean", "label_raw", "id", "split"."""
    out = df.copy()

    # Rename if needed
    if "smiles_clean" not in out.columns and cfg.smiles_col in out.columns:
        out = out.rename(columns={cfg.smiles_col: "smiles_clean"})
    if "label_raw" not in out.columns and cfg.label_col in out.columns:
        out = out.rename(columns={cfg.label_col: "label_raw"})
    if "id" not in out.columns and cfg.id_col in out.columns:
        out = out.rename(columns={cfg.id_col: "id"})

    # Minimal guards
    if "smiles_clean" not in out.columns:
        raise ValueError("Missing 'smiles_clean' (or cfg.smiles_col). Provide cleaned SMILES or set config columns.")
    if "label_raw" not in out.columns:
        raise ValueError("Missing 'label_raw' (or cfg.label_col). Provide labels or set config columns.")

    # Assign ids if needed
    if "id" not in out.columns:
        out["id"] = np.arange(len(out), dtype=np.int64)

    out["split"] = split
    return out


def morgan_fps(smiles_list: List[str], radius: int, n_bits: int) -> List[Optional[DataStructs.ExplicitBitVect]]:
    """Compute Morgan/ECFP fingerprints. Returns None for invalid SMILES."""
    fps: List[Optional[DataStructs.ExplicitBitVect]] = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(None)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(fp)
        except Exception:
            fps.append(None)
    return fps


def _scaffold_fp_for_mol(mol: Optional[Chem.Mol], radius: int, n_bits: int) -> Optional[DataStructs.ExplicitBitVect]:
    if mol is None:
        return None
    try:
        try:
            scaffold = _GraphFramework(mol)
        except Exception:
            scaffold = _GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(scaffold, radius=radius, nBits=n_bits)
    except Exception:
        return None


def scaffold_fps(smiles_list: List[str], radius: int, n_bits: int) -> List[Optional[DataStructs.ExplicitBitVect]]:
    fps: List[Optional[DataStructs.ExplicitBitVect]] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(_scaffold_fp_for_mol(mol, radius, n_bits))
    return fps


def _nn_tanimoto_stats(
    src_fps: List[Optional[DataStructs.ExplicitBitVect]],
    qry_fps: List[Optional[DataStructs.ExplicitBitVect]],
) -> Dict[str, Any]:
    """For each query fp, compute max Tanimoto vs any source fp; return mean/std over queries.
    If no valid comparisons exist, returns {'mean': None, 'std': None, 'n': 0}.
    """
    src_valid = [fp for fp in src_fps if fp is not None]
    if len(src_valid) == 0:
        return {"mean": None, "std": None, "n": 0}

    nn_vals: List[float] = []
    for q in qry_fps:
        if q is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, src_valid)
        if len(sims) == 0:
            continue
        nn_vals.append(max(sims))

    if len(nn_vals) == 0:
        return {"mean": None, "std": None, "n": 0}

    arr = np.asarray(nn_vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()) if len(arr) > 1 else 0.0, "n": int(len(arr))}


def _pairs_above_thresh(
    fpsA: List[Optional[DataStructs.ExplicitBitVect]],
    fpsB: List[Optional[DataStructs.ExplicitBitVect]],
    scafA: List[Optional[DataStructs.ExplicitBitVect]],
    scafB: List[Optional[DataStructs.ExplicitBitVect]],
    smiA: List[str],
    smiB: List[str],
    thr: float,
    intra: bool,
) -> Set[Tuple[int, int]]:
    """Find index pairs (i, j) that are "similar" by MoleculeACE consensus:
    - molecular ECFP Tanimoto >= thr OR
    - scaffold ECFP Tanimoto >= thr OR
    - normalized SMILES Levenshtein similarity >= thr
    Intra => enforce j > i to avoid dup/self.
    """
    pairs: Set[Tuple[int, int]] = set()
    for i, (fa, fsa, sa) in enumerate(zip(fpsA, scafA, smiA)):
        j_start = i + 1 if intra else 0
        if fa is None and fsa is None:
            pass
        for j in range(j_start, len(fpsB)):
            fb, fsb, sb = fpsB[j], scafB[j], smiB[j]

            # skip if both smiles are identical (handled as conflicts elsewhere)
            if intra and sa == sb:
                continue

            tani_ok = False
            leve_ok = False
            scaf_ok = False
           
            if fa is not None and fb is not None:
                tani_ok = DataStructs.TanimotoSimilarity(fa, fb) >= thr
                leve_ok = _levenshtein(sa, sb) >= thr

            if fsa is not None and fsb is not None:
                scaf_ok = DataStructs.TanimotoSimilarity(fsa, fsb) >= thr

            if tani_ok or scaf_ok or leve_ok:
                # print(f"Pair found {tani_ok} {scaf_ok} {leve_ok} {thr} {sa} {sb}")
                pairs.add((i, j))
                
    return pairs


@dataclass(frozen=True)
class StretcherAlignment:
    score: float
    identity_pct: float
    similarity_pct: float
    length: int
    gaps_pct: float
    n_gaps: int
    aligned_query: str
    aligned_subject: str
    query_start: int
    query_end: int
    subject_start: int
    subject_end: int


class PSAStretcherAligner:
    """Thin wrapper around psa.stretcher with caching."""

    def __init__(self):
        if psa is None:
            raise ImportError(
                "pairwise-sequence-alignment is required. "
                "Install EMBOSS (`sudo apt install emboss`) and the Python wrapper "
                "(`pip install pairwise-sequence-alignment`) before running DTI analysis."
            )
        self.moltype = "prot"
        self._cache: Dict[Tuple[str, str], StretcherAlignment] = {}

    def _normalize_seq(self, seq: str) -> str:
        if not isinstance(seq, str):
            seq = "" if pd.isna(seq) else str(seq)
        return "".join(seq.split()).upper()

    def _invert_alignment(self, aln: StretcherAlignment) -> StretcherAlignment:
        return StretcherAlignment(
            score=aln.score,
            identity_pct=aln.identity_pct,
            similarity_pct=aln.similarity_pct,
            length=aln.length,
            gaps_pct=aln.gaps_pct,
            n_gaps=aln.n_gaps,
            aligned_query=aln.aligned_subject,
            aligned_subject=aln.aligned_query,
            query_start=aln.subject_start,
            query_end=aln.subject_end,
            subject_start=aln.query_start,
            subject_end=aln.query_end,
        )

    def _empty_alignment(self, q: str, s: str) -> StretcherAlignment:
        length = max(len(q), len(s))
        return StretcherAlignment(
            score=0.0,
            identity_pct=0.0,
            similarity_pct=0.0,
            length=length,
            gaps_pct=100.0 if length > 0 else 0.0,
            n_gaps=length,
            aligned_query=q or "-" * length,
            aligned_subject=s or "-" * length,
            query_start=1 if q else 0,
            query_end=len(q),
            subject_start=1 if s else 0,
            subject_end=len(s),
        )

    def align(self, query_seq: str, subject_seq: str) -> StretcherAlignment:
        q = self._normalize_seq(query_seq)
        s = self._normalize_seq(subject_seq)
        key = (q, s)
        if key in self._cache:
            return self._cache[key]

        rev_key = (s, q)
        if rev_key in self._cache:
            flipped = self._invert_alignment(self._cache[rev_key])
            self._cache[key] = flipped
            return flipped

        if not q or not s:
            empty_aln = self._empty_alignment(q, s)
            self._cache[key] = empty_aln
            self._cache[rev_key] = self._invert_alignment(empty_aln)
            return empty_aln

        aln = psa.stretcher(
            moltype=self.moltype,
            qseq=q,
            sseq=s,
        )
        result = StretcherAlignment(
            score=float(aln.score),
            identity_pct=float(aln.pidentity),
            similarity_pct=float(getattr(aln, "psimilarity", float("nan"))),
            length=int(aln.length),
            gaps_pct=float(aln.pgaps),
            n_gaps=int(aln.ngaps),
            aligned_query=str(aln.qseq),
            aligned_subject=str(aln.sseq),
            query_start=int(aln.qstart),
            query_end=int(aln.qend),
            subject_start=int(aln.sstart),
            subject_end=int(aln.send),
        )
        self._cache[key] = result
        self._cache[rev_key] = self._invert_alignment(result)
        return result


def _nn_sequence_alignment_stats(
    ref_sequences: Set[str],
    qry_sequences: Set[str],
    aligner: PSAStretcherAligner,
    ref_split: str,
    qry_split: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not ref_sequences or not qry_sequences:
        return (
            {
                "mean_identity_pct": None,
                "std_identity_pct": None,
                "mean_similarity_pct": None,
                "std_similarity_pct": None,
                "mean_score": None,
                "std_score": None,
                "n": 0,
            },
            [],
        )

    ref_list = sorted(ref_sequences)
    qry_list = sorted(qry_sequences)

    identities: List[float] = []
    scores: List[float] = []
    similarities: List[float] = []
    details: List[Dict[str, Any]] = []

    for seq_q in qry_list:
        best_alignment: Optional[StretcherAlignment] = None
        best_seq = None
        for seq_r in ref_list:
            aln = aligner.align(seq_q, seq_r)
            if (
                best_alignment is None
                or aln.identity_pct > best_alignment.identity_pct
                or (
                    aln.identity_pct == best_alignment.identity_pct
                    and aln.score > best_alignment.score
                )
            ):
                best_alignment = aln
                best_seq = seq_r
            if best_alignment is not None and best_alignment.identity_pct >= 99.99:
                break

        if best_alignment is None or best_seq is None:
            continue

        identities.append(float(best_alignment.identity_pct))
        similarities.append(float(best_alignment.similarity_pct))
        scores.append(float(best_alignment.score))
        details.append({
            "split_reference": ref_split,
            "split_query": qry_split,
            "sequence_reference": best_seq,
            "sequence_query": seq_q,
            "identity_pct": float(best_alignment.identity_pct),
            "similarity_pct": float(best_alignment.similarity_pct),
            "score": float(best_alignment.score),
            "alignment_length": int(best_alignment.length),
            "gaps_pct": float(best_alignment.gaps_pct),
            "n_gaps": int(best_alignment.n_gaps),
            "aligned_query": best_alignment.aligned_query,
            "aligned_reference": best_alignment.aligned_subject,
            "query_start": int(best_alignment.query_start),
            "query_end": int(best_alignment.query_end),
            "reference_start": int(best_alignment.subject_start),
            "reference_end": int(best_alignment.subject_end),
        })

    if not identities:
        return (
            {
                "mean_identity_pct": None,
                "std_identity_pct": None,
                "mean_similarity_pct": None,
                "std_similarity_pct": None,
                "mean_score": None,
                "std_score": None,
                "n": 0,
            },
            details,
        )

    arr_id = np.asarray(identities, dtype=float)
    arr_score = np.asarray(scores, dtype=float)
    arr_sim = np.asarray(similarities, dtype=float)
    finite_sim = arr_sim[~np.isnan(arr_sim)]
    if finite_sim.size > 0:
        mean_sim = float(finite_sim.mean())
        std_sim = float(finite_sim.std()) if finite_sim.size > 1 else 0.0
    else:
        mean_sim = None
        std_sim = None
    stats = {
        "mean_identity_pct": float(arr_id.mean()),
        "std_identity_pct": float(arr_id.std()) if len(arr_id) > 1 else 0.0,
        "mean_similarity_pct": mean_sim,
        "std_similarity_pct": std_sim,
        "mean_score": float(arr_score.mean()),
        "std_score": float(arr_score.std()) if len(arr_score) > 1 else 0.0,
        "n": int(len(arr_id)),
    }
    return stats, details


def _compute_sigma3(tv_labels: pd.Series):
    """Return (3 * std, std) for regression labels. Supports scalar or vector labels."""
    if tv_labels.empty:
        return 0.0, 0.0
    if _series_has_sequence_labels(tv_labels):
        matrix = _label_series_to_matrix(tv_labels)
        if matrix.size == 0 or matrix.shape[0] < 2:
            return [0.0] * matrix.shape[1], [0.0] * matrix.shape[1]
        std = np.nanstd(matrix, axis=0, ddof=1)
        sigma3 = (std * 3.0).tolist()
        return sigma3, std.tolist()
    numeric = pd.to_numeric(tv_labels, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0, 0.0
    std = float(np.std(numeric.astype(float), ddof=1))
    return 3.0 * std, std


def _intra_conflict_smiles(df: pd.DataFrame, is_cls: bool, sigma3: Optional[float]) -> Set[str]:
    """Find same-SMILES conflicts within a single split."""
    conflicts: Set[str] = set()
    if df is None or df.empty:
        return conflicts
    for smi, g in df.groupby("smiles_clean"):
        if g.empty:
            continue
        ys = g["label_raw"].tolist()
        if is_cls:
            tuples = [_label_to_tuple(v) for v in ys]
            if len(set(tuples)) > 1:
                conflicts.add(smi)
        else:
            if len(ys) >= 2 and sigma3 is not None:
                found = False
                for i in range(len(ys)):
                    if found:
                        break
                    for j in range(i + 1, len(ys)):
                        delta = _regression_delta(ys[i], ys[j])
                        if _delta_exceeds_threshold(delta, sigma3):
                            conflicts.add(smi)
                            found = True
                            break
    return conflicts


def _cross_conflict_smiles(A: pd.DataFrame, B: pd.DataFrame, is_cls: bool, sigma3: Optional[float]) -> Set[str]:
    """Find same-SMILES conflicts across two splits."""
    merged = A[["smiles_clean", "label_raw"]].merge(
        B[["smiles_clean", "label_raw"]], on="smiles_clean", suffixes=("_A", "_B")
    )
    if merged.empty:
        return set()

    if is_cls:
        tuples_a = merged["label_raw_A"].apply(_label_to_tuple)
        tuples_b = merged["label_raw_B"].apply(_label_to_tuple)
        mask = [a != b for a, b in zip(tuples_a.tolist(), tuples_b.tolist())]
        return set(merged.loc[mask, "smiles_clean"].unique())
    else:
        if sigma3 is None:
            return set()
        mask = [
            _delta_exceeds_threshold(_regression_delta(a, b), sigma3)
            for a, b in zip(merged["label_raw_A"].tolist(), merged["label_raw_B"].tolist())
        ]
        return set(merged.loc[mask, "smiles_clean"].unique())


def _build_conflict_rows(tag: str, smi_set: Set[str], *dfs: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for df in dfs:
        if df.empty:
            continue
        sub = df[df["smiles_clean"].isin(smi_set)]
        for _, r in sub.iterrows():
            rows.append({
                "kind": tag,
                "split": r["split"],
                "id": int(r["id"]),
                "smiles_clean": r["smiles_clean"],
                "label_raw": r["label_raw"],
            })
    return rows


def _cliff_pairs(
    A: pd.DataFrame,
    B: pd.DataFrame,
    fpsA: List[Optional[DataStructs.ExplicitBitVect]],
    fpsB: List[Optional[DataStructs.ExplicitBitVect]],
    scafA: List[Optional[DataStructs.ExplicitBitVect]],
    scafB: List[Optional[DataStructs.ExplicitBitVect]],
    thr: float,
    is_cls: bool,
    sigma3: Optional[float],
    intra: bool,
) -> List[Dict[str, Any]]:
    """Find cliff pairs between A and B and return detailed rows.

    Similarity uses MoleculeACE consensus (see _pairs_above_thresh):
      - molecular ECFP Tanimoto
      - scaffold (generic Murcko) ECFP Tanimoto
      - normalized SMILES Levenshtein similarity

    A row is emitted only if labels differ (classification) or |Δ| ≥ 3σ (regression).
    """
    rows: List[Dict[str, Any]] = []
    smiA = A["smiles_clean"].tolist()
    smiB = B["smiles_clean"].tolist()

    # Candidate similar pairs by consensus (any metric ≥ thr)
    pairs = _pairs_above_thresh(fpsA, fpsB, scafA, scafB, smiA, smiB, thr, intra=intra)

    for i, j in pairs:
        ra, rb = A.iloc[i], B.iloc[j]

        # Decide if this pair is a cliff
        if is_cls:
            labels_a = _label_to_tuple(ra["label_raw"])
            labels_b = _label_to_tuple(rb["label_raw"])
            if labels_a == labels_b:
                continue
            delta = _classification_delta(labels_a, labels_b)
        else:
            if sigma3 is None:
                continue
            delta = _regression_delta(ra["label_raw"], rb["label_raw"])
            if not _delta_exceeds_threshold(delta, sigma3):
                continue

        fa, fb = fpsA[i], fpsB[j]
        fsa, fsb = scafA[i], scafB[j]

        tanimoto = (
            float(DataStructs.TanimotoSimilarity(fa, fb))
            if (fa is not None and fb is not None) else float("nan")
        )
        scaffold_tanimoto = (
            float(DataStructs.TanimotoSimilarity(fsa, fsb))
            if (fsa is not None and fsb is not None) else float("nan")
        )
        levenshtein_sim = float(_levenshtein(ra["smiles_clean"], rb["smiles_clean"]))

        rows.append({
            "kind": "intra" if intra and A is B else "cross",
            "id_A": int(ra["id"]),
            "split_A": ra["split"],
            "smiles_A": ra["smiles_clean"],
            "y_A": ra["label_raw"],
            "id_B": int(rb["id"]),
            "split_B": rb["split"],
            "smiles_B": rb["smiles_clean"],
            "y_B": rb["label_raw"],
            "tanimoto": tanimoto,                 # molecular ECFP Tanimoto
            "scaffold_tanimoto": scaffold_tanimoto,  # generic Murcko ECFP Tanimoto
            "levenshtein_sim": levenshtein_sim,   # SMILES Levenshtein ratio
            "delta": _to_python_scalar(delta),
        })
    return rows


class SMILESAnalyzer:
    """Simplified, modular SMILES analyzer (MoleculeACE-style similarity)."""

    def __init__(self, cfg: AnalyzerConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)

    def _featurize_for_similarity(self, smiles: List[str]):
        """Return tuple of (molecular_fps, scaffold_fps) using same fp settings."""
        mol_fps = morgan_fps(smiles, self.cfg.fp_radius, self.cfg.fp_nbits)
        scaf_fps = scaffold_fps(smiles, self.cfg.fp_radius, self.cfg.fp_nbits)
        return mol_fps, scaf_fps

    def run(self, splits_raw: Dict[str, pd.DataFrame]) -> AnalysisResult:
        self.log.info("Starting SMILES analysis.")

        # Normalize columns + tag split
        splits: Dict[str, pd.DataFrame] = {}
        for split in splits_raw.keys():
            splits[split] = _normalize_columns(splits_raw[split], self.cfg, split)
            self.log.info("Split %s: n=%d", split, len(splits[split]))

        try:
            train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]
            tv_df = pd.concat([train_df, valid_df], ignore_index=True)
        except KeyError:
            train_df, test_df = splits["train"], splits["test"]
            tv_df = train_df
            valid_df = None
        is_cls = (self.cfg.task_type == "classification")

        # Hygiene: duplicates/contamination + REOS stats if present
        all_smiles = tv_df["smiles_clean"].tolist() + test_df["smiles_clean"].tolist()
        n_dup = len(all_smiles) - len(set(all_smiles))
        contaminated = set(tv_df["smiles_clean"]) & set(test_df["smiles_clean"])

        if "n_reos_warnings" in tv_df.columns:
            reos_mean = float(pd.to_numeric(tv_df["n_reos_warnings"], errors="coerce").fillna(0).mean())
            reos_std = float(pd.to_numeric(tv_df["n_reos_warnings"], errors="coerce").fillna(0).std(ddof=1)) if len(tv_df) > 1 else 0.0
        else:
            reos_mean, reos_std = 0.0, 0.0

        self.log.info(
            "Hygiene: duplicates=%d contaminated=%d reos_mean=%.3f reos_std=%.3f",
            n_dup, len(contaminated), reos_mean, reos_std,
        )

        # Statistics
        label_series = tv_df["label_raw"]
        if _series_has_sequence_labels(label_series):
            label_matrix = _label_series_to_matrix(label_series)
            if label_matrix.size == 0:
                tv_mean: Any = []
                base_std: Any = []
            else:
                tv_mean = np.nanmean(label_matrix, axis=0).tolist()
                if label_matrix.shape[0] > 1:
                    base_std = np.nanstd(label_matrix, axis=0, ddof=1).tolist()
                else:
                    base_std = [0.0] * label_matrix.shape[1]
        else:
            numeric = pd.to_numeric(label_series, errors="coerce")
            numeric_clean = numeric.dropna()
            if numeric_clean.empty:
                tv_mean = None
                base_std = 0.0
            else:
                mean_val = numeric_clean.mean()
                tv_mean = float(mean_val) if pd.notna(mean_val) else None
                base_std = float(numeric_clean.std(ddof=1)) if len(numeric_clean) > 1 else 0.0
        if not is_cls:
            sigma3, std_vals = _compute_sigma3(label_series)
            tv_std = std_vals
            self.log.info("Label stats: mean=%s std=%s 3σ=%s", tv_mean, tv_std, sigma3)
        else:
            sigma3 = None
            tv_std = base_std
        
        mol_tv_size = [
            rdMolDescriptors.CalcExactMolWt(
                Chem.MolFromSmiles(smi)
            ) for smi in tv_df["smiles_clean"]
        ]
        mol_tv_size_mean = np.mean(mol_tv_size)
        mol_tv_size_std = np.std(mol_tv_size)

        # Features for similarity (TV and Test): molecular + scaffold
        fps_tv_mol, fps_tv_scaf = self._featurize_for_similarity(tv_df["smiles_clean"].tolist())
        fps_te_mol, fps_te_scaf = self._featurize_for_similarity(test_df["smiles_clean"].tolist())

        # Also featurize individual splits for NN similarity reports
        fps_tr_mol, _ = self._featurize_for_similarity(train_df["smiles_clean"].tolist())
        if valid_df is not None:
            fps_va_mol, _ = self._featurize_for_similarity(valid_df["smiles_clean"].tolist())
        else:
            fps_va_mol = []
            
        # Nearest-neighbor ECFP Tanimoto similarity summaries
        sim_valid_to_train = _nn_tanimoto_stats(fps_tr_mol, fps_va_mol) if valid_df is not None else None
        sim_test_to_train = _nn_tanimoto_stats(fps_tr_mol, fps_te_mol)
        sim_test_to_tv    = _nn_tanimoto_stats(fps_tv_mol, fps_te_mol)


        # Conflicts (same cleaned SMILES)
        intra_train = _intra_conflict_smiles(train_df, is_cls, sigma3)
        intra_valid = _intra_conflict_smiles(valid_df, is_cls, sigma3) if valid_df is not None else []
        intra_test = _intra_conflict_smiles(test_df, is_cls, sigma3)

        cross_tv = _cross_conflict_smiles(train_df, valid_df, is_cls, sigma3) if valid_df is not None else []
        cross_tt = _cross_conflict_smiles(train_df, test_df, is_cls, sigma3)
        cross_vt = _cross_conflict_smiles(valid_df, test_df, is_cls, sigma3) if valid_df is not None else []
        severe_tv_test = _cross_conflict_smiles(tv_df, test_df, is_cls, sigma3)

        self.log.info(
            "Conflicts: intra_train=%d intra_valid=%d intra_test=%d cross_tv=%d cross_tt=%d cross_vt=%d severe_tv_test=%d",
            len(intra_train), len(intra_valid), len(intra_test), len(cross_tv), len(cross_tt), len(cross_vt), len(severe_tv_test)
        )

        # Cliffs (consensus similar-but-different molecules with label delta)
        intra_tv_rows = _cliff_pairs(
            tv_df, tv_df,
            fps_tv_mol, fps_tv_mol,
            fps_tv_scaf, fps_tv_scaf,
            self.cfg.sim_threshold, is_cls, sigma3, intra=True
        )
        if self.cfg.typ != "polaris":
            intra_te_rows = _cliff_pairs(
                test_df, test_df,
                fps_te_mol, fps_te_mol,
                fps_te_scaf, fps_te_scaf,
                self.cfg.sim_threshold, is_cls, sigma3, intra=True
            )
            cross_rows = _cliff_pairs(
                tv_df, test_df,
                fps_tv_mol, fps_te_mol,
                fps_tv_scaf, fps_te_scaf,
                self.cfg.sim_threshold, is_cls, sigma3, intra=False
            )
        else:
            intra_te_rows = []
            cross_rows = []

        self.log.info(
            "Cliffs: intra_tv=%d intra_te=%d cross=%d",
            len(intra_tv_rows), len(intra_te_rows), len(cross_rows)
        )

        # 7) Aggregate summary
        summary: Dict[str, Any] = {
            "counts": {
                "train": int(len(train_df)), 
                "valid": int(len(valid_df)) if valid_df is not None else None, 
                "test": int(len(test_df))
            },
            "hygiene": {
                "n_all_valid_smiles": int(len(all_smiles)),
                "n_unique_valid_smiles": int(len(set(all_smiles))),
                "n_duplicate_valid_smiles": int(n_dup),
                "n_contaminated_tv_vs_test": int(len(contaminated)),
                "reos_mean": float(reos_mean),
                "reos_std": float(reos_std),
            },
            "similarity": {
                "valid_to_train": (
                    {"mean": sim_valid_to_train["mean"], "std": sim_valid_to_train["std"], "n": sim_valid_to_train["n"]}
                    if sim_valid_to_train is not None else None
                ),
                "test_to_train": {
                    "mean": sim_test_to_train["mean"], "std": sim_test_to_train["std"], "n": sim_test_to_train["n"],
                },
                "test_to_trainvalid": {
                    "mean": sim_test_to_tv["mean"], "std": sim_test_to_tv["std"], "n": sim_test_to_tv["n"],
                },
            },
            "task": {
                "type": self.cfg.task_type,
                "label_tv_mean": _to_python_scalar(tv_mean),
                "label_tv_std": _to_python_scalar(tv_std),
                "label_tv_3sigma": _to_python_scalar(sigma3) if sigma3 is not None else None,
                "mol_tv_size_mean": float(mol_tv_size_mean),
                "mol_tv_size_std": float(mol_tv_size_std),
            },
            "conflicts": {
                "intra_train": int(len(intra_train)),
                "intra_valid": int(len(intra_valid)),
                "intra_test": int(len(intra_test)),
                "cross_train_valid": int(len(cross_tv)),
                "cross_train_test": int(len(cross_tt)),
                "cross_valid_test": int(len(cross_vt)),
                "severe_trainvalid_test": int(len(severe_tv_test)),
            },
            "cliffs": {
                "intra_train_valid": int(len(intra_tv_rows)),
                "intra_test": int(len(intra_te_rows)) if intra_te_rows != [] else None,
                "cross_tv_test": int(len(cross_rows)) if cross_rows != [] else None,
                "sim_threshold": float(self.cfg.sim_threshold),
            },
        }

        # 8) Per-record table for drill-down (concat in deterministic order)
        per_record = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        # Conflict rows (detailed)
        conflict_rows: List[Dict[str, Any]] = []
        conflict_rows += _build_conflict_rows("intra_train", intra_train, train_df)
        conflict_rows += _build_conflict_rows("intra_valid", intra_valid, valid_df) if valid_df is not None else []
        conflict_rows += _build_conflict_rows("intra_test", intra_test, test_df)
        conflict_rows += _build_conflict_rows("cross_train_valid", cross_tv, train_df, valid_df) if valid_df is not None else []
        conflict_rows += _build_conflict_rows("cross_train_test", cross_tt, train_df, test_df)
        conflict_rows += _build_conflict_rows("cross_valid_test", cross_vt, valid_df, test_df) if valid_df is not None else []
        conflict_rows += _build_conflict_rows("severe_trainvalid_test", severe_tv_test, tv_df, test_df)

        # Cliff rows are already detailed
        cliff_rows: List[Dict[str, Any]] = []
        cliff_rows += [{**r, "kind": "intra_tv"} for r in intra_tv_rows]
        cliff_rows += [{**r, "kind": "intra_test"} for r in intra_te_rows]
        cliff_rows += [{**r, "kind": "cross_tv_test"} for r in cross_rows]

        self.log.info("SMILES analysis complete.")

        return AnalysisResult(
            summary=summary,
            per_record_df=per_record,
            conflicts_rows=conflict_rows,
            cliffs_rows=cliff_rows,
            sequence_alignment_rows=None,
        )


class DTIAnalyzer:
    """Drug–target interaction analysis: combines molecular + sequence hygiene."""

    _SEQ_COLUMN_CANDIDATES = (
        "sequence_aa",
        "sequence",
        "target_sequence",
        "protein_sequence",
        "aa_sequence",
    )

    def __init__(self, cfg: AnalyzerConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)
        self._smiles = SMILESAnalyzer(cfg, self.log)
        self._aligner = PSAStretcherAligner()

    def _prepare_split(self, split: str, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "sequence_aa" not in out.columns:
            candidates: List[str] = []
            if self.cfg.sequence_col:
                candidates.append(self.cfg.sequence_col)
            candidates.extend([c for c in self._SEQ_COLUMN_CANDIDATES if c not in candidates])
            for col in candidates:
                if col in out.columns:
                    out = out.rename(columns={col: "sequence_aa"})
                    break

        if "sequence_aa" not in out.columns:
            raise ValueError(
                f"Missing amino-acid sequence column for split '{split}'. "
                "Provide 'sequence_aa' or set AnalyzerConfig.sequence_col."
            )

        seq_series = out["sequence_aa"].map(lambda x: "" if pd.isna(x) else str(x))
        seq_series = seq_series.str.upper().str.replace(r"\s+", "", regex=True)
        out["sequence_aa"] = seq_series

        if self.cfg.target_id_col and self.cfg.target_id_col in out.columns:
            out = out.rename(columns={self.cfg.target_id_col: "target_id"})

        empty_count = int((out["sequence_aa"] == "").sum())
        if empty_count > 0:
            self.log.warning("Split %s has %d rows with empty target sequences.", split, empty_count)

        return out

    def _build_drug_summary(self, splits: Dict[str, pd.DataFrame], base_summary: Dict[str, Any]) -> Dict[str, Any]:
        train_df = splits["train"]
        valid_df = splits.get("valid")
        test_df = splits["test"]

        unique_counts = {
            "train": int(train_df["smiles_clean"].nunique()),
            "valid": int(valid_df["smiles_clean"].nunique()) if valid_df is not None else None,
            "test": int(test_df["smiles_clean"].nunique()),
        }

        train_set = set(train_df["smiles_clean"])
        valid_set = set(valid_df["smiles_clean"]) if valid_df is not None else set()
        test_set = set(test_df["smiles_clean"])

        shared_counts = {
            "train_valid": int(len(train_set & valid_set)) if valid_df is not None else None,
            "train_test": int(len(train_set & test_set)),
            "valid_test": int(len(valid_set & test_set)) if valid_df is not None else None,
            "trainvalid_test": int(len((train_set | valid_set) & test_set)),
        }

        hygiene_base = base_summary.get("hygiene", {})
        hygiene = {
            "n_all_smiles": int(hygiene_base.get("n_all_valid_smiles", 0)),
            "n_unique_smiles": int(hygiene_base.get("n_unique_valid_smiles", 0)),
            "n_duplicate_smiles": int(hygiene_base.get("n_duplicate_valid_smiles", 0)),
            "n_contaminated_trainvalid_vs_test": int(hygiene_base.get("n_contaminated_tv_vs_test", 0)),
        }

        return {
            "unique_counts": unique_counts,
            "shared_counts": shared_counts,
            "hygiene": hygiene,
            "similarity": base_summary.get("similarity", {}),
        }

    def _analyze_sequences(
        self,
        splits: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        train_df = splits["train"]
        valid_df = splits.get("valid")
        test_df = splits["test"]

        seq_train = set(train_df["sequence_aa"])
        seq_valid = set(valid_df["sequence_aa"]) if valid_df is not None else set()
        seq_test = set(test_df["sequence_aa"])
        seq_train_valid = seq_train | seq_valid

        unique_counts = {
            "train": int(len(seq_train)),
            "valid": int(len(seq_valid)) if valid_df is not None else None,
            "test": int(len(seq_test)),
        }

        shared_counts = {
            "train_valid": int(len(seq_train & seq_valid)) if valid_df is not None else None,
            "train_test": int(len(seq_train & seq_test)),
            "valid_test": int(len(seq_valid & seq_test)) if valid_df is not None else None,
            "trainvalid_test": int(len(seq_train_valid & seq_test)),
        }

        total_sequences = len(train_df) + (len(valid_df) if valid_df is not None else 0) + len(test_df)
        all_sequences: List[str] = train_df["sequence_aa"].tolist()
        if valid_df is not None:
            all_sequences.extend(valid_df["sequence_aa"].tolist())
        all_sequences.extend(test_df["sequence_aa"].tolist())
        total_unique = len(set(all_sequences))

        duplicates_by_split = {
            "train": int(len(train_df) - train_df["sequence_aa"].nunique()),
            "valid": int(len(valid_df) - valid_df["sequence_aa"].nunique()) if valid_df is not None else None,
            "test": int(len(test_df) - test_df["sequence_aa"].nunique()),
        }

        hygiene = {
            "n_all_sequences": int(total_sequences),
            "n_unique_sequences": int(total_unique),
            "n_duplicate_sequences": int(total_sequences - total_unique),
            "n_contaminated_trainvalid_vs_test": int(len(seq_train_valid & seq_test)),
            "n_contaminated_train_valid": int(len(seq_train & seq_valid)) if valid_df is not None else None,
            "duplicate_sequences_by_split": duplicates_by_split,
        }

        similarity: Dict[str, Optional[Dict[str, Any]]] = {}
        alignment_rows: List[Dict[str, Any]] = []

        if valid_df is not None and seq_valid:
            stats_vt, details_vt = _nn_sequence_alignment_stats(seq_train, seq_valid, self._aligner, "train", "valid")
            similarity["valid_to_train"] = stats_vt
            alignment_rows.extend(details_vt)
        else:
            similarity["valid_to_train"] = None

        stats_tt, details_tt = _nn_sequence_alignment_stats(seq_train, seq_test, self._aligner, "train", "test")
        similarity["test_to_train"] = stats_tt
        alignment_rows.extend(details_tt)

        stats_ttv, details_ttv = _nn_sequence_alignment_stats(seq_train_valid, seq_test, self._aligner, "train_valid", "test")
        similarity["test_to_trainvalid"] = stats_ttv
        alignment_rows.extend(details_ttv)

        if alignment_rows:
            alignment_rows = sorted(
                alignment_rows,
                key=lambda r: (r["identity_pct"], r["score"]),
                reverse=True,
            )[:50]

        pair_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        sequence_index: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"splits": set(), "smiles": set(), "rows": []})

        for split_name, df in splits.items():
            for _, row in df.iterrows():
                smi = row["smiles_clean"]
                seq = row["sequence_aa"]

                pair_entry = pair_index.setdefault((smi, seq), {"splits": set(), "rows": []})
                pair_entry["splits"].add(split_name)
                pair_entry["rows"].append((split_name, row))

                seq_entry = sequence_index[seq]
                seq_entry["splits"].add(split_name)
                seq_entry["smiles"].add(smi)
                seq_entry["rows"].append((split_name, row))

        pair_conflicts = {k: v for k, v in pair_index.items() if len(v["splits"]) > 1}
        sequence_multi = {
            seq: data
            for seq, data in sequence_index.items()
            if len(data["splits"]) > 1 and len(data["smiles"]) > 1
        }

        conflict_rows: List[Dict[str, Any]] = []
        for (smi, seq), data in pair_conflicts.items():
            for split_name, row in data["rows"]:
                rec = {
                    "kind": "dti_pair_cross_split",
                    "split": split_name,
                    "id": _to_python_scalar(row["id"]),
                    "smiles_clean": smi,
                    "sequence_aa": seq,
                    "label_raw": _to_python_scalar(row["label_raw"]),
                }
                if "target_id" in row and not pd.isna(row["target_id"]):
                    rec["target_id"] = _to_python_scalar(row["target_id"])
                conflict_rows.append(rec)

        for seq, data in sequence_multi.items():
            for split_name, row in data["rows"]:
                rec = {
                    "kind": "sequence_multi_smiles_cross_split",
                    "split": split_name,
                    "id": _to_python_scalar(row["id"]),
                    "smiles_clean": row["smiles_clean"],
                    "sequence_aa": seq,
                    "label_raw": _to_python_scalar(row["label_raw"]),
                }
                if "target_id" in row and not pd.isna(row["target_id"]):
                    rec["target_id"] = _to_python_scalar(row["target_id"])
                conflict_rows.append(rec)

        seq_multi_examples = sorted(
            (
                {
                    "sequence": seq,
                    "n_unique_smiles": len(data["smiles"]),
                    "splits": sorted(data["splits"]),
                }
                for seq, data in sequence_multi.items()
            ),
            key=lambda item: item["n_unique_smiles"],
            reverse=True,
        )[:5]

        sequence_summary = {
            "unique_counts": unique_counts,
            "shared_counts": shared_counts,
            "hygiene": hygiene,
            "similarity": similarity,
            "conflicts": {
                "cross_split_pairs": int(len(pair_conflicts)),
                "sequence_multi_smiles": int(len(sequence_multi)),
            },
            "examples": {
                "sequence_multi_smiles": seq_multi_examples,
            },
        }

        self.log.info(
            "Target sequences: unique train=%s valid=%s test=%s shared(train-test)=%d shared(tv-test)=%d pair_conflicts=%d",
            unique_counts["train"],
            unique_counts["valid"],
            unique_counts["test"],
            shared_counts["train_test"],
            shared_counts["trainvalid_test"],
            len(pair_conflicts),
        )

        conflict_counts = {
            "pair_conflicts": int(len(pair_conflicts)),
            "sequence_multi_smiles": int(len(sequence_multi)),
        }

        return sequence_summary, conflict_rows, alignment_rows, conflict_counts

    def run(self, splits_raw: Dict[str, pd.DataFrame]) -> AnalysisResult:
        self.log.info("Starting DTI analysis.")

        prepared: Dict[str, pd.DataFrame] = {}
        for split, df in splits_raw.items():
            prepared[split] = self._prepare_split(split, df)

        smiles_result = self._smiles.run(prepared)

        normalized: Dict[str, pd.DataFrame] = {}
        for split, df in prepared.items():
            normalized[split] = _normalize_columns(df, self.cfg, split)

        sequence_summary, seq_conflict_rows, alignment_rows, conflict_counts = self._analyze_sequences(normalized)

        summary = copy.deepcopy(smiles_result.summary)
        summary["drugs"] = self._build_drug_summary(normalized, summary)
        summary["targets"] = sequence_summary

        conflicts_block = summary.setdefault("conflicts", {})
        conflicts_block["cross_split_dti_pairs"] = conflict_counts["pair_conflicts"]
        conflicts_block["sequence_cross_split_multi_smiles"] = conflict_counts["sequence_multi_smiles"]

        combined_conflict_rows = list(smiles_result.conflicts_rows)
        combined_conflict_rows.extend(seq_conflict_rows)

        self.log.info("DTI analysis complete.")

        return AnalysisResult(
            summary=summary,
            per_record_df=smiles_result.per_record_df,
            conflicts_rows=combined_conflict_rows,
            cliffs_rows=smiles_result.cliffs_rows,
            sequence_alignment_rows=alignment_rows,
        )
