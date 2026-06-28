"""v2: budget enforcement (Orchestrator) and multi-agent coordination."""
from __future__ import annotations

from token_router.accounting import Ledger
from token_router.models.base import Usage

from conductor.backends.base import AgentBackend, AssistantTurn, ToolCall
from conductor.coordinator import Coordinator, Job
from conductor.orchestrator import Orchestrator, ledger_cost_usd
from conductor.pricing import make_pricing
from conductor.tools import READONLY_TOOLS, ToolRegistry


class FixedCostBackend(AgentBackend):
    """Deterministic cost per step: completion_tokens at claude-opus-4-8 output
    rate ($25/1M) -> exact, estimate-free cost for budget tests."""

    def __init__(self, *, completion_tokens: int, loops: bool, backend: str = "anthropic"):
        self.name = "claude-opus-4-8"
        self.backend = backend
        self._ct = completion_tokens
        self._loops = loops
        self._i = 0

    def step(self, *, system, messages, tools, max_tokens=1024, temperature=0.0) -> AssistantTurn:
        self._i += 1
        usage = Usage(prompt_tokens=0, completion_tokens=self._ct,
                      model=self.name, backend=self.backend)
        if self._loops:
            return AssistantTurn(text="", tool_calls=[ToolCall(f"c{self._i}", "now", {})],
                                 stop_reason="tool_use", usage=usage)
        return AssistantTurn(text="done", tool_calls=[], stop_reason="end", usage=usage)


def _registry():
    reg = ToolRegistry()
    for t in READONLY_TOOLS:
        reg.register(t)
    return reg


# $/step = completion_tokens * 25 / 1e6.  400 tokens -> $0.01/step.
_PER_STEP = 0.01


def _agent(loops):
    return FixedCostBackend(completion_tokens=400, loops=loops)


# --- Orchestrator budget ceiling -----------------------------------------

def test_budget_zero_runs_nothing(tmp_path):
    res = Orchestrator(_agent(loops=True), _registry(), run_id="b0",
                       trace_dir=str(tmp_path), budget_usd=0.0, max_steps=10).run("go")
    assert res.status == "budget_exceeded"
    assert res.steps == 0


def test_budget_stops_after_n_steps(tmp_path):
    # budget 0.025 with $0.01/step: steps run at spent 0, .01, .02 (<.025), then
    # .03 >= .025 stops -> 3 steps.
    res = Orchestrator(_agent(loops=True), _registry(), run_id="b3",
                       trace_dir=str(tmp_path), budget_usd=0.025, max_steps=10).run("go")
    assert res.status == "budget_exceeded"
    assert res.steps == 3
    assert abs(res.ledger_summary["est_cost_usd"] - 0.03) < 1e-9


def test_no_budget_runs_to_completion(tmp_path):
    res = Orchestrator(_agent(loops=False), _registry(), run_id="bn",
                       trace_dir=str(tmp_path), max_steps=10).run("go")
    assert res.status == "completed"
    assert res.steps == 1


# --- Coordinator: shared global budget across agents ----------------------

def test_coordinator_skips_agents_once_budget_exhausted(tmp_path):
    # 4 single-step agents at $0.01 each, shared budget $0.015:
    # agent0 (spent0<.015) runs ->.01; agent1 (.01<.015) runs ->.02;
    # agent2 (.02>=.015) skipped; agent3 skipped.
    jobs = [Job(label=f"a{i}", backend=_agent(loops=False), registry=_registry(),
                task="go") for i in range(4)]
    result = Coordinator(budget_usd=0.015, trace_dir=str(tmp_path)).run_all(jobs)

    statuses = [o.status for o in result.outcomes]
    assert statuses == ["completed", "completed", "skipped_budget", "skipped_budget"]
    assert len(result.ran) == 2 and len(result.skipped) == 2
    assert abs(result.total_cost_usd - 0.02) < 1e-9
    # the shared per-provider split is present
    assert "anthropic" in result.ledger_summary["by_backend"]


def test_coordinator_no_budget_runs_all(tmp_path):
    jobs = [Job(label=f"a{i}", backend=_agent(loops=False), registry=_registry(),
                task="go") for i in range(3)]
    result = Coordinator(trace_dir=str(tmp_path)).run_all(jobs)
    assert all(o.status == "completed" for o in result.outcomes)
    assert len(result.ran) == 3


def test_coordinator_isolates_job_errors(tmp_path):
    # A raising job must NOT abort the batch; it gets status="error" and the rest run.
    class Boom(AgentBackend):
        name = "claude-opus-4-8"
        backend = "anthropic"
        def step(self, **kw):
            raise RuntimeError("agent exploded")

    jobs = [
        Job(label="ok1", backend=_agent(loops=False), registry=_registry(), task="go"),
        Job(label="boom", backend=Boom(), registry=_registry(), task="go"),
        Job(label="ok2", backend=_agent(loops=False), registry=_registry(), task="go"),
    ]
    result = Coordinator(trace_dir=str(tmp_path)).run_all(jobs)
    statuses = {o.label: o.status for o in result.outcomes}
    assert statuses == {"ok1": "completed", "boom": "error", "ok2": "completed"}
    boom = next(o for o in result.outcomes if o.label == "boom")
    assert boom.error and "agent exploded" in boom.error
    # the batch still produced a combined result + ledger
    assert "anthropic" in result.ledger_summary["by_backend"]


def test_coordinator_run_ids_unique_no_trace_overwrite(tmp_path):
    # Two run_all calls (or two coordinators) must not overwrite each other's traces.
    def jobs():
        return [Job(label="a", backend=_agent(loops=False), registry=_registry(), task="go")]
    r1 = Coordinator(trace_dir=str(tmp_path)).run_all(jobs())
    r2 = Coordinator(trace_dir=str(tmp_path)).run_all(jobs())
    p1 = r1.outcomes[0].result.trace_path
    p2 = r2.outcomes[0].result.trace_path
    assert p1 != p2  # distinct, collision-resistant run_ids


def test_coordinator_owned_ledger_persisted_to_disk(tmp_path):
    import glob
    Coordinator(trace_dir=str(tmp_path)).run_all(
        [Job(label="a", backend=_agent(loops=False), registry=_registry(), task="go")]
    )
    assert glob.glob(str(tmp_path / "ledger-coord-*.jsonl"))  # crash-durable JSONL written


def test_coordinator_shared_ledger_aggregates(tmp_path):
    # Cheap local agents ($0) + a priced one; the split must show both providers.
    jobs = [
        Job(label="local", backend=FixedCostBackend(completion_tokens=400, loops=False, backend="local"),
            registry=_registry(), task="go"),
        Job(label="claude", backend=_agent(loops=False), registry=_registry(), task="go"),
    ]
    result = Coordinator(trace_dir=str(tmp_path)).run_all(jobs)
    bb = result.ledger_summary["by_backend"]
    assert bb["local"]["cost_usd"] == 0.0
    assert bb["anthropic"]["cost_usd"] > 0.0
