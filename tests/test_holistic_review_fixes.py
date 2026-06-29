"""Regression tests for the holistic-review fixes (v0.3.2 -> next)."""
from __future__ import annotations

from token_router.models.base import Usage

from conductor.backends.base import AgentBackend, AssistantTurn, ToolCall
from conductor.coordinator import Coordinator, Job
from conductor.orchestrator import Orchestrator
from conductor.replay import replay_trace
from conductor.tools import READONLY_TOOLS, ToolRegistry


class FixedCostBackend(AgentBackend):
    """Deterministic $/step (400 completion tokens @ claude-opus-4-8 = $0.01/step)."""

    def __init__(self, *, loops: bool, completion_tokens: int = 400, backend: str = "anthropic"):
        self.name = "claude-opus-4-8"
        self.backend = backend
        self._ct = completion_tokens
        self._loops = loops
        self._i = 0

    def step(self, *, system, messages, tools, max_tokens=1024, temperature=0.0):
        self._i += 1
        usage = Usage(prompt_tokens=0, completion_tokens=self._ct, model=self.name, backend=self.backend)
        if self._loops:
            return AssistantTurn(text="", tool_calls=[ToolCall(f"c{self._i}", "now", {})],
                                 stop_reason="tool_use", usage=usage)
        return AssistantTurn(text="done", tool_calls=[], stop_reason="end", usage=usage)


def _reg():
    r = ToolRegistry()
    for t in READONLY_TOOLS:
        r.register(t)
    return r


def _job(label, loops=False):
    return Job(label=label, backend=FixedCostBackend(loops=loops), registry=_reg(), task="go")


# --- #1 (HIGH): second run_all() on one Coordinator must not collide/double-count ---

def test_coordinator_second_run_all_no_collision_no_double_count(tmp_path):
    coord = Coordinator(trace_dir=str(tmp_path))
    r1 = coord.run_all([_job("a")])
    r2 = coord.run_all([_job("a")])
    # distinct trace files (no truncating overwrite)
    assert r1.outcomes[0].result.trace_path != r2.outcomes[0].result.trace_path
    # fresh ledger per batch -> second batch is NOT the sum of both batches
    assert abs(r1.total_cost_usd - 0.01) < 1e-9
    assert abs(r2.total_cost_usd - 0.01) < 1e-9
    assert r2.ledger_summary["calls"] == 1


# --- #3 (MEDIUM): re-running an owned-ledger Orchestrator stays fresh ---

def test_orchestrator_rerun_owned_ledger_is_fresh(tmp_path):
    orch = Orchestrator(FixedCostBackend(loops=False), _reg(), run_id="rr", trace_dir=str(tmp_path))
    r1 = orch.run("go")
    r2 = orch.run("go")
    # not doubled: each run owns a fresh ledger (disk + memory agree)
    assert abs(r1.ledger_summary["est_cost_usd"] - r2.ledger_summary["est_cost_usd"]) < 1e-9
    assert r2.ledger_summary["calls"] == 1
    import json
    rows = [json.loads(l) for l in open(str(tmp_path / "ledger-rr.jsonl"), encoding="utf-8")]
    assert len(rows) == 1  # disk has only this run's row, not accumulated


# --- #2 (MEDIUM): replay of a budget_exceeded run must NOT report match:True ---

def test_replay_budget_exceeded_status_divergence_is_not_a_match(tmp_path):
    orch = Orchestrator(FixedCostBackend(loops=True), _reg(), run_id="bx",
                        trace_dir=str(tmp_path), budget_usd=0.025, max_steps=10)
    res = orch.run("go")
    assert res.status == "budget_exceeded"
    _r, cmp = replay_trace(res.trace_path, run_id="bx-rep", trace_dir=str(tmp_path))
    assert cmp["original_status"] == "budget_exceeded"
    assert cmp["replayed_status"] != "budget_exceeded"   # no budget re-applied in replay
    assert cmp["status_match"] is False
    assert cmp["match"] is False                          # the divergence is surfaced, not hidden


def test_replay_completed_run_still_matches_with_status(tmp_path):
    # the common (completed) path must still report match:True now that status is compared
    orch = Orchestrator(FixedCostBackend(loops=False), _reg(), run_id="ok", trace_dir=str(tmp_path))
    res = orch.run("go")
    assert res.status == "completed"
    _r, cmp = replay_trace(res.trace_path, run_id="ok-rep", trace_dir=str(tmp_path))
    assert cmp["status_match"] is True and cmp["match"] is True


# --- #5 (test gap): Coordinator stops a looping agent MID-JOB at the budget ---

def test_coordinator_mid_job_budget_cutoff(tmp_path):
    job = Job(label="big", backend=FixedCostBackend(loops=True), registry=_reg(),
              task="go", max_steps=10)
    result = Coordinator(budget_usd=0.025, trace_dir=str(tmp_path)).run_all([job])
    o = result.outcomes[0]
    assert o.status == "budget_exceeded"
    assert o.result is not None and o.result.steps == 3   # stopped mid-job, not at max_steps=10
