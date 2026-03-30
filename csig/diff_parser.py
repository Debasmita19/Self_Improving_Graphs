"""
csig.diff_parser
~~~~~~~~~~~~~~~~
Lightweight, rule-based diff utilities for comparing code / text artefacts.

No LLM calls — everything is deterministic and based on Python's stdlib
``difflib``.  The output feeds directly into ``DiffStats`` and the rule-based
classifier.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from csig.schemas import DiffStats


@dataclass
class FileDiffResult:
    """Diff result for a single pair of file contents."""
    filename: str
    lines_added: int = 0
    lines_removed: int = 0
    module_name: str = ""


def _infer_module(filepath: str) -> str:
    """Heuristically derive a module / component name from a file path."""
    p = Path(filepath)
    stem = p.stem.lower()
    parts = [part.lower() for part in p.parts]

    for part in reversed(parts[:-1]):
        if part not in {"src", "lib", ".", ""}:
            return part

    return stem


def diff_strings(
    old: str,
    new: str,
    filename: str = "<inline>",
) -> FileDiffResult:
    """Compare two text blobs line-by-line and return added/removed counts."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    added = 0
    removed = 0

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, old_lines, new_lines
    ).get_opcodes():
        if tag == "replace":
            removed += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1

    return FileDiffResult(
        filename=filename,
        lines_added=added,
        lines_removed=removed,
        module_name=_infer_module(filename),
    )


def diff_files(old_path: str, new_path: str) -> FileDiffResult:
    """Compare two files on disk."""
    old_text = Path(old_path).read_text(encoding="utf-8", errors="replace")
    new_text = Path(new_path).read_text(encoding="utf-8", errors="replace")
    return diff_strings(old_text, new_text, filename=new_path)


def diff_file_sets(
    old_files: Dict[str, str],
    new_files: Dict[str, str],
) -> List[FileDiffResult]:
    """Compare two mappings of ``{filename: contents}``.

    Files present in only one mapping are treated as fully added / removed.
    """
    all_names = sorted(set(old_files) | set(new_files))
    results: List[FileDiffResult] = []
    for name in all_names:
        old_text = old_files.get(name, "")
        new_text = new_files.get(name, "")
        results.append(diff_strings(old_text, new_text, filename=name))
    return results


def aggregate_diff_stats(results: List[FileDiffResult]) -> DiffStats:
    """Roll up a list of per-file diffs into a single ``DiffStats``."""
    return DiffStats(
        lines_added=sum(r.lines_added for r in results),
        lines_removed=sum(r.lines_removed for r in results),
        files_changed=sum(
            1 for r in results if r.lines_added or r.lines_removed
        ),
    )


def modules_from_results(results: List[FileDiffResult]) -> List[str]:
    """Extract unique module names from a set of diff results."""
    seen: set[str] = set()
    modules: List[str] = []
    for r in results:
        if r.module_name and r.module_name not in seen:
            seen.add(r.module_name)
            modules.append(r.module_name)
    return modules
