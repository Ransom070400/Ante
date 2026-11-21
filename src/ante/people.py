"""Optional face-based memory of people, backed by cheta."""

import time
import logging
from pathlib import Path
from collections.abc import Callable

from cheta import Person, Identity

from reachy_mini import ReachyMini
from ante.memory import memory_path_for_instance


logger = logging.getLogger(__name__)

PEOPLE_DIRNAME = "people"
GREET_COOLDOWN_S = 600.0
GREET_TOPIC = "what do they like or care about"


def create_identity(robot: ReachyMini, instance_path: str | Path | None = None) -> Identity:
    """Return a cheta Identity on the robot's camera.

    Voice is left to the conversation loop, which owns the microphone, so people are recognised by face only.
    """
    data_dir = memory_path_for_instance(instance_path).parent / PEOPLE_DIRNAME
    return Identity(app="ante", camera="reachy", data_dir=data_dir, robot=robot, on_prompt=logger.info)


def find_person(identity: Identity, name: str) -> Person:
    """Return the one enrolled person called ``name``; raise LookupError if none or several match."""
    matches = identity.store.find_by_name(name.strip())
    if not matches:
        raise LookupError(f'nobody called "{name}" is enrolled')
    if len(matches) > 1:
        raise LookupError(f'more than one person is called "{name}"; ask which one is meant')
    return matches[0]


class PersonGreeter:
    """Has the robot greet enrolled people by name when they walk up."""

    def __init__(
        self,
        identity: Identity,
        say: Callable[[str], bool],
        cooldown_s: float = GREET_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Register for arrivals; ``say`` returns False when the robot cannot speak right now."""
        self._identity = identity
        self._say = say
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._last_greeted: dict[str, float] = {}
        identity.on_person_arrived(self.on_arrived)

    def on_arrived(self, person: Person) -> None:
        """Greet ``person`` unless they were greeted recently or the robot is busy."""
        now = self._clock()
        last = self._last_greeted.get(person.id)
        if last is not None and now - last < self._cooldown_s:
            return

        instruction = f"{person.name} has just walked up. Greet them by name in one short sentence."
        memories = self._identity.recall(person, GREET_TOPIC, 1)
        if memories:
            instruction += f" If it fits, mention naturally that you remember: {memories[0].text}"
        if self._say(instruction):
            self._last_greeted[person.id] = now
            logger.info("Greeted %s", person.name)
