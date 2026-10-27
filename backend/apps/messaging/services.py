import json
import logging
from dataclasses import dataclass

from django.conf import settings

from apps.interactions.models import Interaction
from apps.memories.models import Memory

logger = logging.getLogger(__name__)


@dataclass
class GeneratedDraft:
    body: str
    provider: str
    context: dict


def build_context(contact, include_history=True):
    memories = list(
        Memory.objects.filter(contact=contact, allow_drafting=True)
        .order_by("created_at")
        .values("id", "category", "content")
    )
    interactions = []
    if include_history:
        interactions = list(
            Interaction.objects.filter(contact=contact, use_for_drafting=True)
            .order_by("-occurred_at")
            .values("id", "kind", "occurred_at", "notes")[:10]
        )
    return {
        "contact": {
            "id": str(contact.id),
            "name": contact.name,
            "relationship": contact.relationship,
        },
        "memories": [{**item, "id": str(item["id"])} for item in memories],
        "interactions": [
            {
                **item,
                "id": str(item["id"]),
                "occurred_at": item["occurred_at"].isoformat(),
            }
            for item in interactions
        ],
    }


def _template_draft(context, purpose, tone, length, instruction):
    name = context["contact"]["name"]
    detail = ""
    remembered = [item["content"].rstrip(".") for item in context["memories"]]
    if len(remembered) == 1:
        detail = f" I was thinking about {remembered[0]}."
    elif remembered:
        detail = (
            f" I was thinking about {', '.join(remembered[:-1])} and {remembered[-1]}."
        )
    purpose_text = {
        "birthday": "Happy birthday!",
        "congratulations": "Congratulations!",
        "gratitude": "I wanted to say thank you.",
        "reconnect": "It has been a while, and I would love to reconnect.",
    }.get(purpose, "I wanted to check in and see how you are doing.")
    ending = (
        " Hope all is well!" if length == "short" else " I would love to hear how things are going."
    )
    direction = (instruction or "").strip()
    if direction and direction.lower() != "manual" and not direction.lower().startswith("mention "):
        direction = direction[0].upper() + direction[1:]
        if direction[-1] not in ".!?":
            direction += "."
        return f"Hi {name}, {direction}{detail}{ending}".strip()
    return f"Hi {name}, {purpose_text}{detail}{ending}".strip()


def generate_draft(contact, purpose, tone, length, instruction="", include_history=True):
    context = build_context(contact, include_history)
    fallback = _template_draft(context, purpose, tone, length, instruction)
    if not settings.OPENAI_API_KEY:
        return GeneratedDraft(fallback, "template", context)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            timeout=8.0,
            max_retries=0,
        )
        system = (
            "Draft a personal outreach message using only facts in the supplied context. "
            "Never invent events, interests, history, or personal details. If context is sparse, "
            "write a natural generic message. Return only the message body. "
            "If 'instruction' is not empty it is the sender's own direction for this message: "
            "follow it closely, build the message around it, and let it override the generic "
            "purpose wording. Still honour the requested tone and length."
        )
        prompt = {
            "purpose": purpose,
            "tone": tone,
            "length": length,
            "instruction": instruction,
            "context": context,
        }
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        body = (response.choices[0].message.content or "").strip()
        if body:
            return GeneratedDraft(body, "openai-compatible", context)
    except Exception:
        logger.exception("AI draft generation failed; falling back to template")

    return GeneratedDraft(fallback, "template-fallback", context)