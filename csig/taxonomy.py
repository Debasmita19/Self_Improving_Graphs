"""
csig.taxonomy
~~~~~~~~~~~~~
First-pass taxonomy of self-improvement modification types.

Each constant is a canonical string used in ModificationDescriptor.mod_types
and by the rule-based classifier.  Keeping them in one place makes it easy to
extend the taxonomy in later stages without grep-hunting across files.
"""

from typing import FrozenSet

# Canonical modification-type labels (Stage 1 taxonomy)
RETRIEVAL_CHANGE: str              = "retrieval_change"
PROMPT_TEMPLATE_CHANGE: str        = "prompt_template_change"
TOOL_SELECTION_CHANGE: str         = "tool_selection_change"
REASONING_STEP_CHANGE: str         = "reasoning_step_change"
MEMORY_UPDATE: str                 = "memory_update"
ERROR_RETRY_LOGIC: str             = "error_retry_logic"
VERIFIER_CHANGE: str               = "verifier_change"
SCHEMA_LINKING_CHANGE: str         = "schema_linking_change"
EXECUTION_GUARDRAIL_CHANGE: str    = "execution_guardrail_change"
DECOMPOSITION_PLANNING_CHANGE: str = "decomposition_planning_change"

ALL_MOD_TYPES: FrozenSet[str] = frozenset({
    RETRIEVAL_CHANGE,
    PROMPT_TEMPLATE_CHANGE,
    TOOL_SELECTION_CHANGE,
    REASONING_STEP_CHANGE,
    MEMORY_UPDATE,
    ERROR_RETRY_LOGIC,
    VERIFIER_CHANGE,
    SCHEMA_LINKING_CHANGE,
    EXECUTION_GUARDRAIL_CHANGE,
    DECOMPOSITION_PLANNING_CHANGE,
})


def is_valid_mod_type(label: str) -> bool:
    """Return True if *label* belongs to the Stage-1 taxonomy."""
    return label in ALL_MOD_TYPES
