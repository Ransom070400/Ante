"""Optional face-based memory of people, backed by cheta."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path

from reachy_mini import ReachyMini
from ante.memory import memory_path_for_instance


if TYPE_CHECKING:
    from cheta import Person, Identity


logger = logging.getLogger(__name__)

PEOPLE_DIRNAME = "people"


def create_identity(robot: ReachyMini, instance_path: str | Path | None = None) -> "Identity | None":
    """Return a cheta Identity on the robot's camera, or None when cheta is not installed.

    Voice is left to the conversation loop, which owns the microphone, so people are recognised by face only.
    """
    try:
        from cheta import Identity
    except ImportError:
        logger.info('cheta is not installed; people memory is disabled. Install it with: pip install "ante[people]"')
        return None

    data_dir = memory_path_for_instance(instance_path).parent / PEOPLE_DIRNAME
    return Identity(app="ante", camera="reachy", data_dir=data_dir, robot=robot, on_prompt=logger.info)


def find_person(identity: "Identity", name: str) -> "Person":
    """Return the one enrolled person called ``name``; raise LookupError if none or several match."""
    matches = identity.store.find_by_name(name.strip())
    if not matches:
        raise LookupError(f'nobody called "{name}" is enrolled')
    if len(matches) > 1:
        raise LookupError(f'more than one person is called "{name}"; ask which one is meant')
    return matches[0]
