API Objects
===========

This page replaces the old generated markdown class inventory with Sphinx-native
autosummary pages.

Core Orchestration
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   run.main
   utils.build_loader
   utils.build_analyzer
   utils.resolve_output_dir
   utils.ResultWriter

Loaders
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   utils.loader.BaseLoader
   utils.loader.TabularLoader
   utils.loader.TDCLoader
   utils.loader.PolarisLoader
   utils.loader.DTILoader

Analysis
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   utils.analysis.AnalyzerConfig
   utils.analysis.AnalysisResult
   utils.analysis.SMILESAnalyzer
   utils.analysis.DTIAnalyzer
   utils.analysis.StretcherAlignment
   utils.analysis.PSAStretcherAligner

Baselines
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   utils.baselines.BaselineParams
   utils.baselines.run_baselines
