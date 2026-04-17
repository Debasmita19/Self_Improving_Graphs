"""
project_drift.window_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manage reference and current windows of ``RuntimeEvent`` objects.

The window manager is intentionally stateful: call ``add_event`` as
events stream in, and the manager keeps the two windows up to date
using a sliding-window policy.

Serialisation support lets you snapshot a fitted reference distribution
and reload it later (e.g. across agent restarts).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from project_drift.config import DriftConfig
from project_drift.schema import RuntimeEvent

logger = logging.getLogger(__name__)


class WindowManager:
    """Sliding-window container for reference and current event streams.

    Parameters
    ----------
    config:
        Controls ``reference_window_size`` and ``current_window_size``.
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self._cfg = config or DriftConfig()
        self._reference: Deque[RuntimeEvent] = deque(
            maxlen=self._cfg.reference_window_size
        )
        self._current: Deque[RuntimeEvent] = deque(
            maxlen=self._cfg.current_window_size
        )
        self._frozen = False

    # -----------------------------------------------------------------
    # Building the reference window
    # -----------------------------------------------------------------

    def add_reference_event(self, event: RuntimeEvent) -> None:
        """Append an event to the reference window.

        Raises ``RuntimeError`` if the reference window has been frozen.
        """
        if self._frozen:
            raise RuntimeError(
                "Reference window is frozen.  Call reset_reference() "
                "to unfreeze, or add to the current window instead."
            )
        self._reference.append(event)

    def add_reference_events(self, events: List[RuntimeEvent]) -> None:
        for e in events:
            self.add_reference_event(e)

    def freeze_reference(self) -> None:
        """Lock the reference window so no more events can be added.

        Call this after populating the reference from historical data,
        before starting the live stream.
        """
        self._frozen = True
        logger.info(
            "Reference window frozen with %d events.", len(self._reference)
        )

    def reset_reference(self) -> None:
        """Clear and unfreeze the reference window."""
        self._reference.clear()
        self._frozen = False

    # -----------------------------------------------------------------
    # Building the current (test) window
    # -----------------------------------------------------------------

    def add_current_event(self, event: RuntimeEvent) -> None:
        """Append an event to the current / live window."""
        self._current.append(event)

    def add_current_events(self, events: List[RuntimeEvent]) -> None:
        for e in events:
            self.add_current_event(e)

    def clear_current(self) -> None:
        """Empty the current window (e.g. after running a drift test)."""
        self._current.clear()

    # -----------------------------------------------------------------
    # Convenience: stream-oriented auto-routing
    # -----------------------------------------------------------------

    def add_event(self, event: RuntimeEvent) -> None:
        """If the reference is frozen, route to current; otherwise to reference."""
        if self._frozen:
            self.add_current_event(event)
        else:
            self.add_reference_event(event)

    # -----------------------------------------------------------------
    # Access
    # -----------------------------------------------------------------

    @property
    def reference_events(self) -> List[RuntimeEvent]:
        return list(self._reference)

    @property
    def current_events(self) -> List[RuntimeEvent]:
        return list(self._current)

    @property
    def n_reference(self) -> int:
        return len(self._reference)

    @property
    def n_current(self) -> int:
        return len(self._current)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def is_ready(self, min_reference: int = 20, min_current: int = 10) -> bool:
        """True if both windows have enough events for a meaningful test."""
        return (
            self.n_reference >= min_reference
            and self.n_current >= min_current
        )

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def save_reference(self, path: str) -> None:
        """Persist the reference window to JSON."""
        data = {
            "frozen": self._frozen,
            "events": [e.to_dict() for e in self._reference],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Reference window saved to %s (%d events).", path, len(self._reference))

    def load_reference(self, path: str, freeze: bool = True) -> None:
        """Load a previously saved reference window.

        Parameters
        ----------
        freeze:
            If ``True`` (default), freeze the window after loading so
            subsequent events go to the current window.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        events = [RuntimeEvent.from_dict(d) for d in raw["events"]]
        self.reset_reference()
        self.add_reference_events(events)
        if freeze:
            self.freeze_reference()
        logger.info(
            "Reference window loaded from %s (%d events, frozen=%s).",
            path, len(events), freeze,
        )
