from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _normalize_fracs(fracs) -> Tuple[float, float, float]:
    """Return (train, valid, test) fractions from list/tuple or dict."""
    if fracs is None:
        return 0.8, 0.1, 0.1
    if isinstance(fracs, dict):
        train = float(fracs.get("train"))
        valid = float(fracs.get("valid"))
        test = float(fracs.get("test"))
    elif isinstance(fracs, (list, tuple)):
        if len(fracs) != 3:
            raise ValueError("split_fracs must have three entries: [train, valid, test]")
        train, valid, test = (float(fracs[0]), float(fracs[1]), float(fracs[2]))
    else:
        raise TypeError("split_fracs must be a list/tuple or dict with train/valid/test")

    total = train + valid + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_fracs must sum to 1.0 (got {total:.6f})")
    return train, valid, test


def _rng(seed: Optional[int]):
    return np.random.RandomState(seed) if seed is not None else np.random


def random_split_indices(
    n_items: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
    seed: Optional[int] = 123,
) -> Tuple[List[int], List[int], List[int]]:
    rng = _rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    train_cut = int(frac_train * n_items)
    valid_cut = int((frac_train + frac_valid) * n_items)
    train_idx = indices[:train_cut].tolist()
    valid_idx = indices[train_cut:valid_cut].tolist()
    test_idx = indices[valid_cut:].tolist()
    return train_idx, valid_idx, test_idx


def _scaffold_smiles(smiles: str, include_chirality: bool = False) -> Optional[str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split_indices(
    smiles_list: Iterable[str],
    frac_train: float,
    frac_valid: float,
    frac_test: float,
    seed: Optional[int] = 123,
) -> Tuple[List[int], List[int], List[int]]:
    """DeepChem-style scaffold split: group by Bemis-Murcko scaffold, then size-sort."""
    scaffolds: Dict[Optional[str], List[int]] = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold = _scaffold_smiles(smiles)
        scaffolds.setdefault(scaffold, []).append(idx)

    # DeepChem-style: sort by scaffold set size (descending), use optional shuffle for ties.
    items = list(scaffolds.items())
    if seed is not None:
        rng = _rng(seed)
        rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    n_items = sum(len(v) for _, v in items)
    train_cut = frac_train * n_items
    valid_cut = (frac_train + frac_valid) * n_items

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in items:
        if len(train_idx) + len(idxs) <= train_cut:
            train_idx.extend(idxs)
        elif len(train_idx) + len(valid_idx) + len(idxs) <= valid_cut:
            valid_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return train_idx, valid_idx, test_idx


def split_indices(
    smiles_list: Iterable[str],
    method: str,
    fracs,
    seed: Optional[int] = 123,
) -> Tuple[List[int], List[int], List[int]]:
    frac_train, frac_valid, frac_test = _normalize_fracs(fracs)
    method_norm = method.strip().lower()
    smiles_list = list(smiles_list)
    if method_norm in {"random", "rand"}:
        return random_split_indices(len(smiles_list), frac_train, frac_valid, frac_test, seed=seed)
    if method_norm in {"scaffold", "scaffold_split"}:
        return scaffold_split_indices(smiles_list, frac_train, frac_valid, frac_test, seed=seed)
    raise ValueError(f"Unknown splitter method: {method}")
