"""Shared OpenRouter / Groq LLM client for intent classification and adjudication."""

from __future__ import annotations

from . import config


def make_client():
    """Returns an OpenAI client pointed at OpenRouter."""
    from openai import OpenAI

    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/Kcodess2807/graphsight",
            "X-Title": "Graphsight",
        },
    )


def make_intent_client() -> tuple[object, str]:
    """Returns (client, model) for intent classification. Prefers Groq for speed."""
    if config.GROQ_API_KEY:
        from openai import OpenAI

        client = OpenAI(
            base_url=config.GROQ_BASE_URL,
            api_key=config.GROQ_API_KEY,
        )
        return client, config.GROQ_MODEL
    return make_client(), config.OPENROUTER_MODEL
