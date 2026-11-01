"""Corpus importers (DIME, ESLORA, DIHANA) with strict label/coverage semantics.

Everything here is import/verification infrastructure: it never creates labels,
never turns missing labels into negatives and never relaxes the human-annotation
contract of ``mrcd.dataset.build_recording_windows``.
"""

from mrcd.corpora.base import (
    Annotation,
    CorpusFormatError,
    Example,
    ForbiddenMappingError,
    InvalidBoundsError,
    LeakError,
    SourceFile,
    UnsafeArchiveError,
    assert_features_clean,
    find_response_tokens,
    ms_to_s,
)
from mrcd.corpora.groups import (
    AmbiguousGroupError,
    PartitionLeakError,
    UnresolvedGroupError,
    assert_no_group_crossing,
    find_duplicate_sources,
    find_group_crossings,
    group_for_name,
    load_sample_parents,
)

__all__ = [
    "AmbiguousGroupError",
    "Annotation",
    "CorpusFormatError",
    "Example",
    "ForbiddenMappingError",
    "InvalidBoundsError",
    "LeakError",
    "PartitionLeakError",
    "SourceFile",
    "UnresolvedGroupError",
    "UnsafeArchiveError",
    "assert_features_clean",
    "assert_no_group_crossing",
    "find_duplicate_sources",
    "find_group_crossings",
    "find_response_tokens",
    "group_for_name",
    "load_sample_parents",
    "ms_to_s",
]
