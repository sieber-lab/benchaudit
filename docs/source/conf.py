from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "BenchAudit"
author = "BenchAudit contributors"
copyright = "2026, BenchAudit contributors"
release = "0.1.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Keep docs builds lightweight by mocking optional scientific deps.
autodoc_mock_imports = [
    "Levenshtein",
    "chembl_structure_pipeline",
    "lightgbm",
    "matplotlib",
    "numpy",
    "pandas",
    "polaris",
    "psa",
    "pystow",
    "rdkit",
    "rdkit.Chem",
    "rdkit.Chem.AllChem",
    "rdkit.Chem.MolStandardize",
    "rdkit.Chem.MolStandardize.rdMolStandardize",
    "rdkit.Chem.Scaffolds",
    "rdkit.Chem.Scaffolds.MurckoScaffold",
    "rdkit.Chem.rdchem",
    "rdkit.DataStructs",
    "scipy",
    "scipy.stats",
    "seaborn",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.metrics",
    "sklearn.neural_network",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "tdc",
    "tdc.single_pred",
    "torch",
    "tqdm",
    "tqdm.auto",
    "useful_rdkit_utils",
    "xgboost",
    "yaml",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"
html_title = "BenchAudit Documentation"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
}

modindex_common_prefix = ["utils."]

# Allow local "make html" even if users run from a virtualenv with strict hash checking.
os.environ.setdefault("SPHINX_DISABLE_REDIRECTS", "1")
