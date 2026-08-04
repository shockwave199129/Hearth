"""Regression tests for the first-run setup completion flag.

These exist because of a specific near-miss. `orchestrator.py` re-exports
`mark_setup_complete` under the name `mark_complete`, and main.py calls it
as `orchestrator.mark_complete()` — attribute access through a module
namespace, which Pyflakes cannot see. A lint autofix deleted the import as
"unused", which would have made every successful setup raise AttributeError
at the final step, *after* packages and models were already installed: the
most expensive possible place to fail, and invisible until someone ran a
real first-run install on a clean machine.

Nothing in the suite covered it. These tests are that coverage.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup import orchestrator, state


@pytest.fixture
def isolated_setup_db(tmp_path, monkeypatch):
    """Point the setup flag at a throwaway DB so these never touch the
    developer's real profile.db. get_connection() creates the full schema
    on connect, so no separate migration step is needed here."""
    monkeypatch.setattr(state, "SETUP_STATE_DB_PATH", tmp_path / "profile.db")
    return tmp_path


def test_orchestrator_still_exposes_mark_complete():
    """The exact breakage described in the module docstring.

    main.py calls `orchestrator.mark_complete()`; assert the attribute is
    present AND is the real implementation, so neither deleting the import
    nor shadowing it with a stub passes."""
    assert hasattr(orchestrator, "mark_complete"), (
        "main.py calls orchestrator.mark_complete() — the re-export in "
        "orchestrator.py was removed. See that file's noqa comment."
    )
    assert orchestrator.mark_complete is state.mark_setup_complete


def test_setup_facade_names_used_by_main_are_all_present():
    """main.py drives setup entirely through this module, so the whole
    surface it reaches for is pinned here rather than one name at a time."""
    for name in ("detect_status", "run_setup", "mark_complete", "clear_setup_complete"):
        assert hasattr(orchestrator, name), f"orchestrator.{name} is gone"


def test_flag_round_trips(isolated_setup_db):
    """Fresh DB reads False, mark makes it True, clear makes it False —
    the contract detect_status() relies on at orchestrator.py's
    `flagged = is_setup_complete()` branch."""
    assert state.is_setup_complete() is False

    state.mark_setup_complete()
    assert state.is_setup_complete() is True

    state.clear_setup_complete()
    assert state.is_setup_complete() is False


def test_mark_complete_is_idempotent(isolated_setup_db):
    """Setup can legitimately run twice (the UI offers Retry, and
    /api/setup/start is documented as idempotent), so the second mark must
    upsert rather than raise on the id=1 primary key."""
    state.mark_setup_complete()
    state.mark_setup_complete()
    assert state.is_setup_complete() is True


def test_mark_complete_via_orchestrator_alias_writes_the_flag(isolated_setup_db):
    """End-to-end through the name main.py actually uses — a re-export that
    exists but points somewhere inert would still pass the assertions
    above."""
    orchestrator.mark_complete()
    assert state.is_setup_complete() is True
