"""Tests for the people memory tools."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ante.tools.core_tools import ToolDependencies
from ante.tools.who_is_here import WhoIsHere
from ante.tools.enroll_person import EnrollPerson
from ante.tools.forget_person import ForgetPerson
from ante.tools.remember_about_person import RememberAboutPerson


ALICE = SimpleNamespace(id="p1", name="Alice", confidence=0.84)


def _deps(identity: Any = None) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_enabled=True, identity=identity
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (WhoIsHere(), {}),
        (EnrollPerson(), {"name": "Alice", "consent": True}),
        (RememberAboutPerson(), {"name": "Alice", "fact": "Likes tea"}),
        (ForgetPerson(), {"name": "Alice", "confirmed": True}),
    ],
)
async def test_tools_degrade_without_cheta(tool: Any, kwargs: dict[str, Any]) -> None:
    """Without an identity every tool returns an error instead of raising into the loop."""
    result = await tool(_deps(), **kwargs)
    assert "error" in result


@pytest.mark.asyncio
async def test_who_is_here_returns_name_and_memories() -> None:
    """A recognised person comes back with the facts remembered about them."""
    identity = MagicMock()
    identity.who_is_here.return_value = ALICE
    identity.recall.return_value = [MagicMock(text="Is learning the cello")]

    result = await WhoIsHere()(_deps(identity), topic="music")

    assert result == {"recognized": True, "name": "Alice", "confidence": 0.84, "memories": ["Is learning the cello"]}
    identity.recall.assert_called_once_with(ALICE, "music", 5)


@pytest.mark.asyncio
async def test_who_is_here_reports_an_unknown_face() -> None:
    """Nobody recognised is a normal result, not an error."""
    identity = MagicMock()
    identity.who_is_here.return_value = None

    assert await WhoIsHere()(_deps(identity)) == {"recognized": False}


@pytest.mark.asyncio
async def test_enroll_person_refuses_without_consent() -> None:
    """Nothing is enrolled unless the model says the person consented."""
    identity = MagicMock()

    result = await EnrollPerson()(_deps(identity), name="Alice", consent=False)

    assert "error" in result
    identity.enroll.assert_not_called()


@pytest.mark.asyncio
async def test_enroll_person_enrolls_by_face_only() -> None:
    """Voice is skipped because the conversation loop owns the microphone."""
    identity = MagicMock()
    identity.enroll.return_value = ALICE

    result = await EnrollPerson()(_deps(identity), name="Alice", consent=True)

    assert result == {"enrolled": "Alice"}
    identity.enroll.assert_called_once_with("Alice", consent=True, voice=False)


@pytest.mark.asyncio
async def test_remember_about_person_saves_fact_for_named_person() -> None:
    """The fact is stored against the person found by name, with the optional expiry."""
    identity = MagicMock()
    identity.store.find_by_name.return_value = [ALICE]

    result = await RememberAboutPerson()(_deps(identity), name="Alice", fact="Has a trip on Friday", expires_in_days=7)

    assert result == {"saved": "Has a trip on Friday", "person": "Alice"}
    identity.remember.assert_called_once_with(ALICE, "Has a trip on Friday", 7)


@pytest.mark.asyncio
async def test_remember_about_person_reports_unknown_name() -> None:
    """An unenrolled name is reported so the model can offer to enroll them."""
    identity = MagicMock()
    identity.store.find_by_name.return_value = []

    result = await RememberAboutPerson()(_deps(identity), name="Bob", fact="Likes tea")

    assert "enrolled" in result["error"]
    identity.remember.assert_not_called()


@pytest.mark.asyncio
async def test_forget_person_requires_confirmation() -> None:
    """Deletion is irreversible, so an unconfirmed call deletes nothing."""
    identity = MagicMock()
    identity.store.find_by_name.return_value = [ALICE]

    result = await ForgetPerson()(_deps(identity), name="Alice", confirmed=False)

    assert "error" in result
    identity.forget.assert_not_called()


@pytest.mark.asyncio
async def test_forget_person_deletes_confirmed_person() -> None:
    """A confirmed request hard-deletes the named person."""
    identity = MagicMock()
    identity.store.find_by_name.return_value = [ALICE]

    result = await ForgetPerson()(_deps(identity), name="Alice", confirmed=True)

    assert result == {"forgotten": "Alice"}
    identity.forget.assert_called_once_with(ALICE)
