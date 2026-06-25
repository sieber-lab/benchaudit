API Objects
===========

This page is an object inventory only. Detailed API documentation lives in the
module pages below to avoid duplicate object descriptions.

Core Orchestration (``run`` and ``utils``)
------------------------------------------

- ``run.main``
- ``utils.build_loader``
- ``utils.build_analyzer``
- ``utils.resolve_output_dir``
- ``utils.ResultWriter``

See :doc:`api_core` for full docs.

Loaders (``utils.loader``)
--------------------------

- ``utils.loader.BaseLoader``
- ``utils.loader.TabularLoader``
- ``utils.loader.TDCLoader``
- ``utils.loader.PolarisLoader``
- ``utils.loader.DTILoader``

See :doc:`api_loaders` for full docs.

Analysis and Baselines (``utils.analysis`` and ``utils.baselines``)
-------------------------------------------------------------------

- ``utils.analysis.AnalyzerConfig``
- ``utils.analysis.AnalysisResult``
- ``utils.analysis.SMILESAnalyzer``
- ``utils.analysis.DTIAnalyzer``
- ``utils.analysis.StretcherAlignment``
- ``utils.analysis.PSAStretcherAligner``
- ``utils.baselines.BaselineParams``
- ``utils.baselines.run_baselines``

See :doc:`api_analysis` for full docs.

Rank-Fragility Analysis (``utils.rank_fragility``)
--------------------------------------------------

- ``utils.rank_fragility.config.RunConfig``
- ``utils.rank_fragility.audit.audit_dataset``
- ``utils.rank_fragility.panels.generate_counterfactual_panels``
- ``utils.rank_fragility.counterfactual.run_counterfactual_evaluation``
- ``utils.rank_fragility.fragility.compute_fragility_summary``

See :doc:`api_rank_fragility` for full docs.

Support modules are documented in :doc:`api_support`.
