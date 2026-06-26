from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from helpers.prepare_lp_pdbbind import OUTPUT_COLUMNS, prepare_dataframe, prepare_lp_pdbbind


def _fixture_frame() -> pd.DataFrame:
    rows = [
        {
            "pdbid": "1aaa",
            "header": "hydrolase",
            "smiles": "CCO",
            "category": "refined",
            "seq": "AAAA",
            "resolution": 1.2,
            "date": "2020-01-01",
            "type": "hydrolase",
            "new_split": "train",
            "CL1": True,
            "CL2": True,
            "CL3": False,
            "remove_for_balancing_val": False,
            "kd/ki": "Kd=1nM",
            "value": 9.0,
            "covalent": False,
        },
        {
            "pdbid": "2bbb",
            "header": "transferase",
            "smiles": "CCN",
            "category": "core",
            "seq": "BBBB",
            "resolution": 2.1,
            "date": "2020-01-02",
            "type": "transferase",
            "new_split": "val",
            "CL1": True,
            "CL2": False,
            "CL3": False,
            "remove_for_balancing_val": True,
            "kd/ki": "Ki=2nM",
            "value": "8.7",
            "covalent": False,
        },
        {
            "pdbid": "3ccc",
            "header": "lyase",
            "smiles": "CCC",
            "category": "general",
            "seq": "CCCC",
            "resolution": 1.8,
            "date": "2020-01-03",
            "type": "lyase",
            "new_split": "test",
            "CL1": True,
            "CL2": True,
            "CL3": True,
            "remove_for_balancing_val": False,
            "kd/ki": "Kd=3nM",
            "value": 8.5,
            "covalent": False,
        },
        {
            "pdbid": "4ddd",
            "header": "unsplit",
            "smiles": "CCCl",
            "category": "general",
            "seq": "DDDD",
            "resolution": 2.0,
            "date": "2020-01-04",
            "type": "other",
            "new_split": pd.NA,
            "CL1": True,
            "CL2": True,
            "CL3": True,
            "remove_for_balancing_val": False,
            "kd/ki": "Kd=4nM",
            "value": 8.4,
            "covalent": False,
        },
        {
            "pdbid": "5eee",
            "header": "covalent",
            "smiles": "CCBr",
            "category": "general",
            "seq": "EEEE",
            "resolution": 2.0,
            "date": "2020-01-05",
            "type": "other",
            "new_split": "test",
            "CL1": True,
            "CL2": True,
            "CL3": True,
            "remove_for_balancing_val": False,
            "kd/ki": "Kd=5nM",
            "value": 8.3,
            "covalent": True,
        },
        {
            "pdbid": "6fff",
            "header": "unclean",
            "smiles": "CCF",
            "category": "general",
            "seq": "FFFF",
            "resolution": 2.0,
            "date": "2020-01-06",
            "type": "other",
            "new_split": "train",
            "CL1": False,
            "CL2": False,
            "CL3": False,
            "remove_for_balancing_val": False,
            "kd/ki": "Kd=6nM",
            "value": 8.2,
            "covalent": False,
        },
    ]
    return pd.DataFrame(rows).set_index("pdbid")


class PrepareLPPDBBindTests(unittest.TestCase):
    def test_prepare_dataframe_filters_and_normalizes_splits(self) -> None:
        prepared = prepare_dataframe(_fixture_frame())

        self.assertEqual(prepared["_split"].value_counts().to_dict(), {"train": 1, "valid": 1, "test": 1})
        self.assertEqual(prepared["ID"].tolist(), ["id_1aaa", "id_2bbb", "id_3ccc"])
        self.assertEqual(prepared.loc[prepared["_split"] == "valid", "new_split"].tolist(), ["val"])
        self.assertEqual(prepared.loc[prepared["_split"] == "valid", "binding_measure"].tolist(), ["Ki=2nM"])

    def test_prepare_lp_pdbbind_writes_splits_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "LP_PDBBind.csv"
            out_dir = root / "prepared"
            _fixture_frame().to_csv(source)

            manifest = prepare_lp_pdbbind(
                source=str(source),
                out_dir=out_dir,
                check_expected_counts=False,
            )

            self.assertEqual(manifest["split_counts"], {"train": 1, "valid": 1, "test": 1})
            self.assertTrue((out_dir / "manifest.json").exists())
            valid = pd.read_csv(out_dir / "valid.csv")
            self.assertEqual(list(valid.columns), OUTPUT_COLUMNS)
            self.assertEqual(valid["pdbid"].tolist(), ["2bbb"])
            self.assertEqual(valid["remove_for_balancing_val"].tolist(), [True])


if __name__ == "__main__":
    unittest.main()
