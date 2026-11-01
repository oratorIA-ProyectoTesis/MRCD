"""MRCD service surface: contract v1 (schema + validator), file-in/JSON-out analyzer and a local dev server.

Nothing here is imported by the engine; the engine does not depend on this package.
"""

from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer, build_analysis, error_document
from mrcd.service.contract import ANALYSIS_SCHEMA_VERSION, JOB_SCHEMA_VERSION, SCHEMA, validate_document

__all__ = [
    "AnalysisOptions",
    "MrcdAnalyzer",
    "build_analysis",
    "error_document",
    "ANALYSIS_SCHEMA_VERSION",
    "JOB_SCHEMA_VERSION",
    "SCHEMA",
    "validate_document",
]
