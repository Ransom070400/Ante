import asyncio
import logging
from typing import Any

from ante.people import find_person
from ante.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ForgetPerson(Tool):
    """Permanently delete everything remembered about an enrolled person."""

    name = "forget_person"
    description = (
        "Permanently delete an enrolled person: their face signature and every fact remembered about them. "
        "Call this when they ask to be forgotten. This cannot be undone, so confirm out loud first and set "
        "confirmed to true only after they say yes."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the enrolled person to forget."},
            "confirmed": {"type": "boolean", "description": "True only after the person confirmed the deletion."},
        },
        "required": ["name", "confirmed"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Hard-delete the named person."""
        if deps.identity is None:
            return {"error": "People memory is not available on this robot"}

        name = kwargs.get("name")
        if not isinstance(name, str) or not name.strip():
            return {"error": "name must be a non-empty string"}
        if kwargs.get("confirmed") is not True:
            return {"error": "deletion is not confirmed; ask the person first"}

        identity = deps.identity
        try:
            person = find_person(identity, name)
            await asyncio.to_thread(identity.forget, person)
        except LookupError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.warning("forget_person failed: %s", exc)
            return {"error": f"could not forget the person: {type(exc).__name__}: {exc}"}

        logger.info("Tool call: forget_person name=%s", person.name)
        return {"forgotten": person.name}
