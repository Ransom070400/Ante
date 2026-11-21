import asyncio
import logging
from typing import Any

from ante.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class EnrollPerson(Tool):
    """Learn a new person's face so the robot can recognise them later."""

    name = "enroll_person"
    description = (
        "Learn the face of the person in front of the robot so you can recognise them and remember things about "
        "them later. ONLY call this after the person has clearly agreed out loud to being remembered; set consent "
        "to true only then. Before calling, tell them to look at the camera and slowly turn their head side to "
        "side. It takes several seconds. Only a numeric face signature is stored, never a photo."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name the person asked to be called."},
            "consent": {
                "type": "boolean",
                "description": "True only if the person explicitly agreed to be remembered.",
            },
        },
        "required": ["name", "consent"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Enroll the visible person under the given name."""
        if deps.identity is None:
            return {"error": "People memory is not available on this robot"}
        if not deps.camera_enabled:
            return {"error": "Camera is disabled"}

        name = kwargs.get("name")
        if not isinstance(name, str) or not name.strip():
            return {"error": "name must be a non-empty string"}
        if kwargs.get("consent") is not True:
            return {"error": "the person has not consented; ask them first"}

        identity = deps.identity
        try:
            person = await asyncio.to_thread(lambda: identity.enroll(name, consent=True, voice=False))
        except Exception as exc:
            logger.warning("enroll_person failed: %s", exc)
            return {"error": f"enrollment failed: {type(exc).__name__}: {exc}"}

        logger.info("Tool call: enroll_person name=%s", person.name)
        return {"enrolled": person.name}
