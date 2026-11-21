import asyncio
import logging
from typing import Any

from ante.people import find_person
from ante.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RememberAboutPerson(Tool):
    """Save one short fact about an enrolled person."""

    name = "remember_about_person"
    description = (
        "Save ONE short fact about a specific enrolled person so you can bring it up when you see them again. "
        "Use it for details they shared: family, pets, hobbies, work, plans, birthdays. Do not save sensitive data "
        "(passwords, addresses, payment info, health diagnoses). The person must already be enrolled with "
        "enroll_person. For facts about whoever you are talking to without enrolling them, use remember instead. "
        "Do it silently; acknowledge naturally."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the enrolled person the fact is about."},
            "fact": {
                "type": "string",
                "description": 'A short third-person statement, such as "Is learning the cello". One fact per call.',
            },
            "expires_in_days": {
                "type": "integer",
                "description": "Forget the fact after this many days. Use for temporary plans. Omit for lasting facts.",
            },
        },
        "required": ["name", "fact"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Store one fact about the named person."""
        if deps.identity is None:
            return {"error": "People memory is not available on this robot"}

        name = kwargs.get("name")
        fact = kwargs.get("fact")
        expires_in_days = kwargs.get("expires_in_days")
        if not isinstance(name, str) or not name.strip():
            return {"error": "name must be a non-empty string"}
        if not isinstance(fact, str) or not fact.strip():
            return {"error": "fact must be a non-empty string"}
        if expires_in_days is not None and (not isinstance(expires_in_days, int) or expires_in_days <= 0):
            return {"error": "expires_in_days must be a positive integer"}

        identity = deps.identity
        try:
            person = find_person(identity, name)
            await asyncio.to_thread(identity.remember, person, fact, expires_in_days)
        except LookupError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.warning("remember_about_person failed: %s", exc)
            return {"error": f"could not save the fact: {type(exc).__name__}: {exc}"}

        logger.info("Tool call: remember_about_person name=%s fact=%s", person.name, fact[:120])
        return {"saved": fact.strip(), "person": person.name}
