"""
csig.classifier
~~~~~~~~~~~~~~~
Rule-based heuristic classifier that infers modification types from
filenames, folder names, changed-module names, and keyword substrings.

No LLM calls — deterministic pattern matching only.
"""

from __future__ import annotations

from typing import List, Sequence

from csig import taxonomy as tx


# Each rule is a (substring_or_keyword, mod_type) pair.  If the substring
# appears in *any* of the input signals (lowercased), the mod_type is emitted.
_KEYWORD_RULES: list[tuple[str, str]] = [
    ("prompt",        tx.PROMPT_TEMPLATE_CHANGE),
    ("template",      tx.PROMPT_TEMPLATE_CHANGE),
    ("retriev",       tx.RETRIEVAL_CHANGE),
    ("rag",           tx.RETRIEVAL_CHANGE),
    ("search",        tx.RETRIEVAL_CHANGE),
    ("tool",          tx.TOOL_SELECTION_CHANGE),
    ("router",        tx.TOOL_SELECTION_CHANGE),
    ("dispatch",      tx.TOOL_SELECTION_CHANGE),
    ("reason",        tx.REASONING_STEP_CHANGE),
    ("chain_of",      tx.REASONING_STEP_CHANGE),
    ("cot",           tx.REASONING_STEP_CHANGE),
    ("memory",        tx.MEMORY_UPDATE),
    ("cache",         tx.MEMORY_UPDATE),
    ("buffer",        tx.MEMORY_UPDATE),
    ("retry",         tx.ERROR_RETRY_LOGIC),
    ("error",         tx.ERROR_RETRY_LOGIC),
    ("fallback",      tx.ERROR_RETRY_LOGIC),
    ("verif",         tx.VERIFIER_CHANGE),
    ("check",         tx.VERIFIER_CHANGE),
    ("valid",         tx.VERIFIER_CHANGE),
    ("schema",        tx.SCHEMA_LINKING_CHANGE),
    ("link",          tx.SCHEMA_LINKING_CHANGE),
    ("guardrail",     tx.EXECUTION_GUARDRAIL_CHANGE),
    ("guard",         tx.EXECUTION_GUARDRAIL_CHANGE),
    ("sandbox",       tx.EXECUTION_GUARDRAIL_CHANGE),
    ("decompos",      tx.DECOMPOSITION_PLANNING_CHANGE),
    ("plan",          tx.DECOMPOSITION_PLANNING_CHANGE),
    ("subtask",       tx.DECOMPOSITION_PLANNING_CHANGE),
]


def classify_mod_types(
    *,
    filenames: Sequence[str] = (),
    modules: Sequence[str] = (),
    keywords: Sequence[str] = (),
    rationale: str = "",
) -> List[str]:
    """Infer zero or more taxonomy labels from textual signals.

    Parameters
    ----------
    filenames : sequence of str
        Changed file paths / names.
    modules : sequence of str
        Inferred module or component names.
    keywords : sequence of str
        Arbitrary keywords (e.g. from commit messages).
    rationale : str
        Free-text rationale string to scan for clues.

    Returns
    -------
    list[str]
        Deduplicated, deterministically ordered list of mod-type labels.
    """
    corpus = " ".join(
        list(filenames) + list(modules) + list(keywords) + [rationale]
    ).lower()

    seen: set[str] = set()
    result: list[str] = []
    for keyword, mod_type in _KEYWORD_RULES:
        if keyword in corpus and mod_type not in seen:
            seen.add(mod_type)
            result.append(mod_type)
    return result
