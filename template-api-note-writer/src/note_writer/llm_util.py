import os
import requests

###########################################################################
# Basic Anthropic wrapper
###########################################################################

_API_URL = "https://api.anthropic.com/v1/messages"
_HEADERS = {
    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


def _anthropic_chat(messages, *, model="claude-3-7-sonnet-20250219",
                    temperature=0.7, max_tokens=1024):
    """
    Minimal, blocking call to Anthropic Messages API.
    Returns *plain text* (first block of assistant content).
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(_API_URL, headers=_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


###########################################################################
# Public helpers (Claude‑native names) ‑‑ keep signatures identical to Grok
###########################################################################

def get_claude_response(prompt: str, temperature: float = 0.7) -> str:
    """Single‑shot chat completion."""
    return _anthropic_chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )


def claude_describe_image(image_url: str, temperature: float = 0.0) -> str:
    """
    Vision call (Claude 3 only).  **Requires** vision access in your Anthropic console.
    """
    return _anthropic_chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "url": image_url},
                    {
                        "type": "text",
                        "content": "Describe the image accurately and objectively.",
                    },
                ],
            }
        ],
        model="claude-3-opus-20240229",   # any Claude‑3 model with vision works
        max_tokens=256,
        temperature=temperature,
    )


def get_claude_live_search_response(prompt: str, temperature: float = 0.7) -> str:
    """
    Claude doesn’t have built‑in web search like Grok; we simply ask Claude.
    For production‑grade retrieval use a real search API (SerpAPI, Google CSE, etc.)
    and feed results back into the prompt.
    """
    return get_claude_response(prompt, temperature)


###########################################################################
# Legacy Aliases — so the rest of the codebase *just works*
###########################################################################

get_grok_response = get_claude_response
grok_describe_image = claude_describe_image
get_grok_live_search_response = get_claude_live_search_response
