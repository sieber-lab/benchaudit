from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run


class HelperFunctionTests(unittest.TestCase):
    def test_echo_config_only_keeps_known_keys(self) -> None:
        cfg = {
            "type": "tabular",
            "name": "demo",
            "task": "classification",
            "modality": "tabular",
            "seed": 7,
            "out": "runs",
            "extra": "ignored",
        }

        echo = run.echo_config(cfg)
        self.assertEqual(
            set(echo.keys()),
            {"type", "name", "task", "modality", "seed", "out"},
        )

    def test_load_yaml_reads_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.yaml"
            path.write_text("type: tabular\nname: tiny\n", encoding="utf-8")
            data = run.load_yaml(path)

        self.assertEqual(data["type"], "tabular")
        self.assertEqual(data["name"], "tiny")

    def test_load_yaml_rejects_empty_document(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                run.load_yaml(path)

    def test_load_yaml_rejects_non_mapping_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "list.yaml"
            path.write_text("- a\n- b\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                run.load_yaml(path)


if __name__ == "__main__":
    unittest.main()
