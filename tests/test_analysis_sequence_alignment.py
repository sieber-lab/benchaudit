from __future__ import annotations

import shutil
import unittest
from unittest.mock import Mock

from utils import _build_analyzer_config
from utils.analysis import (
    AnalyzerConfig,
    PSAStretcherAligner,
    StretcherAlignment,
    _nn_sequence_alignment_stats,
)


class _FakeAligner:
    def align(self, query_seq: str, subject_seq: str) -> StretcherAlignment:
        length = max(len(query_seq), len(subject_seq))
        matches = sum(a == b for a, b in zip(query_seq, subject_seq))
        identity_pct = (matches / length * 100.0) if length else 0.0
        return StretcherAlignment(
            score=float(matches),
            identity_pct=identity_pct,
            similarity_pct=identity_pct,
            length=length,
            gaps_pct=0.0,
            n_gaps=0,
            aligned_query=query_seq,
            aligned_subject=subject_seq,
            query_start=1 if query_seq else 0,
            query_end=len(query_seq),
            subject_start=1 if subject_seq else 0,
            subject_end=len(subject_seq),
        )


class SequenceAlignmentTests(unittest.TestCase):
    def test_sequence_alignment_workers_parsed_from_info(self) -> None:
        cfg = _build_analyzer_config(
            {
                "type": "dti",
                "modality": "dti",
                "task": "classification",
                "info": {"sequence_alignment_workers": "4"},
            }
        )

        self.assertEqual(cfg.sequence_alignment_workers, 4)

    def test_analyzer_config_rejects_invalid_sequence_alignment_workers(self) -> None:
        with self.assertRaises(Exception) as ctx:
            AnalyzerConfig(
                task_type="classification",
                typ="tabular",
                sequence_alignment_workers=0,
            )
        self.assertIn("sequence_alignment_workers", str(ctx.exception))

    def test_parallel_nn_sequence_alignment_matches_serial(self) -> None:
        ref_sequences = {"AAAA", "AAAB", "CCCC"}
        qry_sequences = {"AAAA", "AAAC", "CCCA"}

        serial_stats, serial_rows = _nn_sequence_alignment_stats(
            ref_sequences,
            qry_sequences,
            _FakeAligner(),
            "train",
            "test",
            workers=1,
        )
        parallel_stats, parallel_rows = _nn_sequence_alignment_stats(
            ref_sequences,
            qry_sequences,
            _FakeAligner(),
            "train",
            "test",
            workers=3,
        )

        self.assertEqual(serial_stats, parallel_stats)
        self.assertEqual(serial_rows, parallel_rows)

    def test_nn_sequence_alignment_logs_progress(self) -> None:
        logger = Mock()

        _nn_sequence_alignment_stats(
            {"AAAA", "AAAB"},
            {"AAAA", "AAAC"},
            _FakeAligner(),
            "train",
            "test",
            workers=2,
            progress_logger=logger,
        )

        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertTrue(any("Sequence alignment %s -> %s:" in msg for msg in messages))
        self.assertTrue(any("progress" in msg for msg in messages))
        self.assertTrue(any("complete" in msg for msg in messages))

    @unittest.skipUnless(shutil.which("stretcher"), "EMBOSS stretcher is not installed")
    def test_direct_emboss_stretcher_aligner_runs(self) -> None:
        aligner = PSAStretcherAligner()

        aln = aligner.align("AAAA", "AAAT")

        self.assertEqual(aln.length, 4)
        self.assertEqual(aln.identity_pct, 75.0)
        self.assertEqual(aln.aligned_query, "AAAA")
        self.assertEqual(aln.aligned_subject, "AAAT")


if __name__ == "__main__":
    unittest.main()
