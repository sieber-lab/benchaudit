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

Support modules are documented in :doc:`api_support`.
