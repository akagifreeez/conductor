"""Per-provider pricing for the cost ledger.

Reuses token-router's ``Pricing`` (USD per 1,000,000 tokens; ``backend=="local"``
-> $0) and seeds it with a Conductor price table. Local inference (Ollama / LM
Studio, labeled backend ``"local"``) is free by construction; remote providers
are priced by model id.

Honesty note: the Claude figures are from Anthropic's published
per-token pricing; the OpenAI-family figures are approximate and should be
confirmed against the provider's current pricing page before quoting real costs
- ``Pricing.override(...)`` makes that a one-liner.

About the unknown-model fallback: any model id NOT in the table below (including
the OpenRouter / Groq / Mistral / Together models reachable via the OpenAI-compat
adapter) is priced by token-router's inherited placeholder rate ($0.50/$0.50 per
1M). That guarantees the cost is **never silently $0** - but it is a fixed
placeholder, NOT a real estimate: it will OVER-state a cheaper provider and
UNDER-state a premium one. Treat any cost for an unlisted model as a rough
placeholder and ``override(...)`` it before quoting. (The honest framing matters
here precisely because per-provider cost is a headline feature.)
"""
from __future__ import annotations

from token_router.pricing import Pricing

# model id -> (prompt_usd_per_1M, completion_usd_per_1M)
CONDUCTOR_PRICES = {
    # Anthropic (published per-token pricing).
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
    # OpenAI family (APPROXIMATE - confirm on the pricing page before quoting).
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def make_pricing() -> Pricing:
    """A ``Pricing`` seeded with Conductor's table (local stays $0)."""
    return Pricing(prices=CONDUCTOR_PRICES)
