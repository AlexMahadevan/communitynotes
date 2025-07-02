import json
import os
from typing import List

from data_models import MisleadingTag, Post
from note_writer.llm_util import get_grok_response   # Claude alias

# ────────────────────────────────────────────────────────────
# Optional feature flag so you can skip this step altogether
# (set DISABLE_MISLEADING_TAGS=true in the workflow env)
# ────────────────────────────────────────────────────────────
DISABLE_MISLEADING_TAGS = os.getenv("DISABLE_MISLEADING_TAGS", "false").lower() == "true"

TAG_SET = {
    "factual_error",
    "manipulated_media",
    "outdated_information",
    "missing_important_context",
    "disputed_claim_as_fact",
    "misinterpreted_satire",
    "other",
}

PROMPT_TEMPLATE = """Below is an X post and a proposed Community Note.
Return JSON *only* with one key "misleading_tags" whose value is a list of
one‑word tags from the allowed list.

ALLOWED TAGS:
{tags}

POST TEXT:
{post_text}

{images_block}
PROPOSED NOTE (or refusal):
{note_text}
"""

def get_misleading_tags(
    post: Post,
    images_summary: str,
    note_text: str,
    retries: int = 3,
) -> List[MisleadingTag]:
    """
    Return a *deduplicated* list of MisleadingTag enums or [].
    Will never raise unless every retry fails to return valid JSON.
    """

    if DISABLE_MISLEADING_TAGS or note_text.startswith("NO NOTE NEEDED"):
        return []

    prompt = _build_prompt(post, images_summary, note_text)

    while retries > 0:
        try:
            raw = get_grok_response(prompt)
            parsed = json.loads(raw)

            tags = parsed.get("misleading_tags", [])
            tags = [t.strip().lower() for t in tags if t.strip().lower() in TAG_SET]
            return [MisleadingTag(tag) for tag in dict.fromkeys(tags)]   # dedupe

        except Exception as e:
            print(f"[warn] bad misleading_tags JSON → {e}. LLM said: {raw[:120]!r}")
            retries -= 1

    # If we reach here, we tried 'retries' times and still couldn't parse.
    print("[warn] giving up on misleading_tags for this post.")
    return []      # soft‑fail instead of raising

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _build_prompt(post: Post, images_summary: str, note_text: str) -> str:
    images_block = (
        f"IMAGE SUMMARY:\n{images_summary or 'None'}"
        if images_summary else "No images."
    )
    return PROMPT_TEMPLATE.format(
        tags="\n".join(f"- {t}" for t in TAG_SET),
        post_text=post.text,
        images_block=images_block,
        note_text=note_text,
    )
