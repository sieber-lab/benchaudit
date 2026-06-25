"""Chemistry helpers used by rank-fragility audits."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

LOG = logging.getLogger(__name__)

rdBase.DisableLog("rdApp.warning")


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string without raising on invalid input."""
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception as exc:  # pragma: no cover - defensive RDKit guard
        LOG.debug("failed to parse SMILES %r: %s", smiles, exc)
        return None


def standardize_smiles(
    smiles: str,
    keep_largest_fragment: bool = True,
    neutralize: bool = True,
    canonicalize_tautomer: bool = False,
    keep_stereo: bool = True,
) -> str | None:
    """Return a canonical RDKit SMILES string, or None for invalid molecules."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        if keep_largest_fragment:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        if neutralize:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if canonicalize_tautomer:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_stereo)
    except Exception as exc:
        LOG.debug("failed to standardize SMILES %r: %s", smiles, exc)
        return None


def murcko_scaffold_smiles(smiles: str) -> str | None:
    """Return canonical Bemis-Murcko scaffold SMILES, or None if unavailable."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        smi = Chem.MolToSmiles(scaffold, canonical=True)
        return smi or None
    except Exception as exc:
        LOG.debug("failed to compute scaffold for %r: %s", smiles, exc)
        return None


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Return an ECFP/Morgan bit vector, or None for invalid molecules."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception as exc:
        LOG.debug("failed to fingerprint %r: %s", smiles, exc)
        return None


def max_tanimoto_to_train(test_fps: Iterable, train_fps: Iterable) -> np.ndarray:
    """Compute each test fingerprint's maximum Tanimoto similarity to train."""
    test_fps = list(test_fps)
    train_valid = [fp for fp in train_fps if fp is not None]
    out: list[float] = []
    if not train_valid:
        return np.full(len(test_fps), np.nan, dtype=float)

    for fp in test_fps:
        if fp is None:
            out.append(np.nan)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_valid)
        out.append(float(max(sims)) if sims else np.nan)
    return np.asarray(out, dtype=float)
