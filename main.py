"""Module entry point for the BenchAudit command line interface."""

from __future__ import annotations

from run import main as run_main


def main() -> None:
    """Run the same command line interface exposed by ``run.py``."""
    run_main()


if __name__ == "__main__":
    main()
