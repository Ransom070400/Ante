import asyncio
import logging
from typing import Any

from ante.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

RECALL_LIMIT = 5


class WhoIsHere(Tool):
    """Recognise the person in front of the camera and recall what is remembered about them."""

    name = "who_is_here"
    description = (
        "Look through the camera and recognise who is in front of the robot by their face, then return their name "
        "and the facts remembered about them. Use this when the user asks who they are or whether you know them, "
        "or when someone new arrives and you want to greet them by name. It can take several seconds and the person "
        "may need to turn their head slowly so the robot can tell they are not a photo. Returns no name when the "
        "person is not enrolled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "What to recall about them, such as hobbies, family, or work. Defaults to anything.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Recognise the visible person and recall facts about them."""
        if deps.identity is None:
            return {"error": "People memory is not available on this robot"}
        if not deps.camera_enabled:
            return {"error": "Camera is disabled"}

        topic = kwargs.get("topic")
        query = topic.strip() if isinstance(topic, str) and topic.strip() else "what do they like or care about"
        identity = deps.identity
        try:
            person = await asyncio.to_thread(identity.who_is_here)
            if person is None:
                logger.info("Tool call: who_is_here nobody recognised")
                return {"recognized": False}
            memories = await asyncio.to_thread(identity.recall, person, query, RECALL_LIMIT)
        except Exception as exc:
            logger.warning("who_is_here failed: %s", exc)
            return {"error": f"recognition failed: {type(exc).__name__}: {exc}"}

        logger.info("Tool call: who_is_here recognised=%s", person.name)
        return {
            "recognized": True,
            "name": person.name,
            "confidence": round(person.confidence, 2),
            "memories": [memory.text for memory in memories],
        }
