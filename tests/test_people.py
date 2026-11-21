"""Tests for the optional cheta-backed people memory."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ante.people import find_person, create_identity


def _identity_with(*people: SimpleNamespace) -> MagicMock:
    identity = MagicMock()
    identity.store.find_by_name.return_value = list(people)
    return identity


def test_find_person_returns_the_single_match() -> None:
    """One enrolled person with that name is returned."""
    alice = SimpleNamespace(id="p1", name="Alice")
    assert find_person(_identity_with(alice), " Alice ") is alice


@pytest.mark.parametrize("matches", [(), (SimpleNamespace(id="p1"), SimpleNamespace(id="p2"))])
def test_find_person_rejects_missing_or_ambiguous_names(matches: tuple[SimpleNamespace, ...]) -> None:
    """No match and several matches both raise, so a tool never acts on the wrong person."""
    with pytest.raises(LookupError):
        find_person(_identity_with(*matches), "Sam")


def test_create_identity_is_disabled_without_cheta(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional dependency disables people memory instead of failing startup."""
    monkeypatch.setitem(sys.modules, "cheta", None)
    assert create_identity(MagicMock()) is None
