# Bench Setup Notes

## Sequence Alignment Dependencies
- Install the EMBOSS toolkit so that `stretcher`/`needle` binaries are available: `sudo apt install emboss`.
- Install the Python wrapper that drives these programs: `pip install pairwise-sequence-alignment`.
- No additional configuration is required; the analyzer invokes `psa.stretcher` directly once both pieces are present.

## DTI Configuration Hints
- Set `modality: dti` in your YAML config to enable the DTI analyzer.
- Provide sequence metadata under `info`, e.g. `sequence_col` for the amino-acid column and (optionally) `target_id_col` for target identifiers.
- For tabular DTI data, keep `info.keep_invalid: True` (default) so SMILES cleaning preserves row alignment with sequence fields.
