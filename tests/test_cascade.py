"""Cost cascade: cheap-first, escalate-on-low-confidence, both costs ledgered."""
from __future__ import annotations

from token_router.accounting import Ledger
from token_router.models.base import Usage

from conductor.backends.base import AssistantTurn, Message
from conductor.backends.cascade import CascadeBackend, default_gate
from conductor.backends.scripted import ScriptedBackend, ScriptedTurn
from conductor.orchestrator import Orchestrator
from conductor.pricing import make_pricing
from conductor.tools import READONLY_TOOLS, ToolRegistry


def _usage(backend="x"):
    return Usage(prompt_tokens=1, completion_tokens=1, model="m", backend=backend)


def _turn(text="", tool_calls=None):
    return AssistantTurn(text=text, tool_calls=tool_calls or [], stop_reason="end", usage=_usage())


# --- default gate heuristic ----------------------------------------------

def test_gate_keeps_tool_calls_turn():
    from conductor.backends.base import ToolCall
    assert default_gate(_turn(tool_calls=[ToolCall("c", "now", {})])) is True


def test_gate_keeps_substantial_answer():
    assert default_gate(_turn(text="The capital of France is Paris.")) is True


def test_gate_escalates_short_answer():
    assert default_gate(_turn(text="ok")) is False


def test_gate_escalates_hedging_answer():
    assert default_gate(_turn(text="I'm not sure, I don't know the answer to that.")) is False


# --- cascade routing ------------------------------------------------------

class _ExplodingBackend(ScriptedBackend):
    def step(self, **kw):
        raise AssertionError("strong backend must not be called when cheap is accepted")


def test_no_escalation_when_cheap_is_confident():
    cheap = ScriptedBackend([ScriptedTurn(text="A clear, sufficient answer to the question.")],
                            name="local-m", backend="local")
    strong = _ExplodingBackend([ScriptedTurn(text="x")], name="claude", backend="anthropic")
    casc = CascadeBackend(cheap, strong)
    turn = casc.step(system="s", messages=[Message(role="user", text="q")], tools=[])
    assert turn.usage.backend == "local"
    assert turn.extra_usages == []


def test_escalation_carries_cheap_cost():
    cheap = ScriptedBackend([ScriptedTurn(text="idk")], name="local-m", backend="local")
    strong = ScriptedBackend([ScriptedTurn(text="A full, confident answer from the strong model.")],
                             name="claude-opus-4-8", backend="anthropic")
    casc = CascadeBackend(cheap, strong)
    turn = casc.step(system="s", messages=[Message(role="user", text="q")], tools=[])
    assert turn.usage.backend == "anthropic"          # used the strong model
    assert len(turn.extra_usages) == 1
    assert turn.extra_usages[0].backend == "local"    # cheap attempt's cost retained


def test_orchestrator_ledger_records_both_on_escalation(tmp_path):
    reg = ToolRegistry()
    for t in READONLY_TOOLS:
        reg.register(t)
    cheap = ScriptedBackend([ScriptedTurn(text="no")], name="local-m", backend="local")
    strong = ScriptedBackend([ScriptedTurn(text="A complete answer from the strong model here.")],
                             name="claude-opus-4-8", backend="anthropic")
    led = Ledger(pricing=make_pricing())
    res = Orchestrator(CascadeBackend(cheap, strong), reg, run_id="csc",
                       trace_dir=str(tmp_path), ledger=led).run("hard question")
    s = res.ledger_summary
    assert set(s["by_backend"]) == {"local", "anthropic"}   # both providers billed
    assert s["by_backend"]["anthropic"]["cost_usd"] > 0
    assert s["by_backend"]["local"]["cost_usd"] == 0.0
