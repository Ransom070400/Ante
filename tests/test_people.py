"""Tests for the optional cheta-backed people memory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ante.people import PersonGreeter, find_person


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


def _greeter(say_result: bool = True, now: list[float] | None = None) -> tuple[PersonGreeter, MagicMock, MagicMock]:
    identity = MagicMock()
    identity.recall.return_value = [SimpleNamespace(text="Is learning the cello")]
    say = MagicMock(return_value=say_result)
    clock = now if now is not None else [0.0]
    return PersonGreeter(identity, say, cooldown_s=60.0, clock=lambda: clock[0]), identity, say


def test_greeter_greets_by_name_with_a_remembered_fact() -> None:
    """An arrival is greeted by name and the most relevant memory is offered."""
    greeter, identity, say = _greeter()

    greeter.on_arrived(SimpleNamespace(id="p1", name="Alice"))

    identity.on_person_arrived.assert_called_once_with(greeter.on_arrived)
    spoken = say.call_args.args[0]
    assert "Alice" in spoken and "Is learning the cello" in spoken


def test_greeter_waits_out_the_cooldown() -> None:
    """The same person is not greeted again until the cooldown has passed."""
    now = [0.0]
    greeter, _, say = _greeter(now=now)
    alice = SimpleNamespace(id="p1", name="Alice")

    greeter.on_arrived(alice)
    now[0] = 30.0
    greeter.on_arrived(alice)
    assert say.call_count == 1

    now[0] = 61.0
    greeter.on_arrived(alice)
    assert say.call_count == 2


def test_greeter_retries_when_the_robot_could_not_speak() -> None:
    """A greeting the robot was too busy to say does not start the cooldown."""
    greeter, _, say = _greeter(say_result=False)
    alice = SimpleNamespace(id="p1", name="Alice")

    greeter.on_arrived(alice)
    greeter.on_arrived(alice)

    assert say.call_count == 2
