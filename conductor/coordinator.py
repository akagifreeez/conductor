"""Multi-agent coordination under one shared cost ceiling (v2).

A control plane should be able to run *several* agents and guarantee their
combined spend stays under a budget - not just bound each agent in isolation.
The Coordinator runs a list of agent jobs against a SINGLE shared ledger with a
SINGLE global ``budget_usd``:

* each job runs through a normal ``Orchestrator`` that shares the ledger, so the
  budget it checks between steps is the *global* spend so far;
* before starting a job, if the budget is already exhausted, the job is skipped
  (``status="skipped_budget"``) rather than started;
* the combined result reports per-job status + the shared per-provider ledger.

This reuses the existing ledger and orchestrator verbatim - the only new idea is
"one budget, many agents". Jobs run sequentially for deterministic accounting
(the shared ledger is not designed for concurrent writers).

Honest bound: the ceiling stops *further* work; a single in-flight step can
overshoot by its own cost, so the final total may exceed ``budget_usd`` by at
most the most-expensive single step. The Coordinator never silently hides that.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from token_router.accounting import Ledger

from .backends.base import AgentBackend
from .orchestrator import (
    DEFAULT_SYSTEM,
    Orchestrator,
    RunResult,
    conductor_summary,
    ledger_cost_usd,
)
from .pricing import make_pricing
from .tools.registry import ToolRegistry


@dataclass
class Job:
    """One agent task: a label, a backend, a registry, and the task text."""

    label: str
    backend: AgentBackend
    registry: ToolRegistry
    task: str
    system: Optional[str] = None
    max_steps: int = 8


@dataclass
class JobOutcome:
    label: str
    status: str                       # RunResult status | "skipped_budget" | "error"
    result: Optional[RunResult] = None
    error: Optional[str] = None       # set when the job raised (status="error")


@dataclass
class CoordinatorResult:
    outcomes: List[JobOutcome] = field(default_factory=list)
    total_cost_usd: float = 0.0
    budget_usd: Optional[float] = None
    ledger_summary: dict = field(default_factory=dict)

    @property
    def ran(self) -> List[JobOutcome]:
        return [o for o in self.outcomes if o.status != "skipped_budget"]

    @property
    def skipped(self) -> List[JobOutcome]:
        return [o for o in self.outcomes if o.status == "skipped_budget"]


class Coordinator:
    """Runs many agent jobs under one shared ledger + global budget."""

    def __init__(
        self,
        *,
        budget_usd: Optional[float] = None,
        trace_dir: str = "traces",
        ledger: Optional[Ledger] = None,
        run_id_prefix: str = "coord",
    ) -> None:
        self.budget_usd = budget_usd
        self.trace_dir = trace_dir
        self.run_id_prefix = run_id_prefix
        # A caller-owned (shared) ledger persists across batches; an owned ledger
        # is created FRESH per run_all() (see below), never once in __init__ - so
        # repeated run_all() calls don't collide on filenames or double-count cost.
        self._injected_ledger = ledger
        self._batch = 0
        # Exposed after run_all so callers can inspect the most recent batch ledger.
        self.ledger = ledger if ledger is not None else Ledger(pricing=make_pricing())

    def run_all(self, jobs: List[Job]) -> CoordinatorResult:
        result = CoordinatorResult(budget_usd=self.budget_usd)
        self._batch += 1
        # Fresh per-CALL token so repeated run_all() invocations never collide on
        # trace/ledger filenames (Tracer truncates) - a per-Coordinator token did.
        batch = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}-b{self._batch}"
        owns_ledger = self._injected_ledger is None
        if owns_ledger:
            # Fresh owned ledger per batch: crash-durable JSONL, and no stale rows
            # from a prior batch leaking into this batch's budget/summary.
            path = os.path.join(self.trace_dir, f"ledger-{self.run_id_prefix}-{batch}.jsonl")
            ledger = Ledger(pricing=make_pricing(), jsonl_path=path)
        else:
            ledger = self._injected_ledger
        self.ledger = ledger
        try:
            for i, job in enumerate(jobs):
                # Skip before starting if the global budget is already spent.
                if self.budget_usd is not None and ledger_cost_usd(ledger) >= self.budget_usd:
                    result.outcomes.append(JobOutcome(label=job.label, status="skipped_budget"))
                    continue
                orch = Orchestrator(
                    job.backend,
                    job.registry,
                    run_id=f"{self.run_id_prefix}-{batch}-{i}-{job.label}",
                    system=job.system or DEFAULT_SYSTEM,
                    max_steps=job.max_steps,
                    trace_dir=self.trace_dir,
                    ledger=ledger,               # shared -> global budget for this batch
                    budget_usd=self.budget_usd,  # each step also checks the global spend
                )
                # Isolate per-job failures: one erroring agent must not abort the
                # batch or discard the combined result. The Orchestrator re-raises
                # on a backend failure, so we catch it here and record status="error".
                try:
                    run = orch.run(job.task)
                    result.outcomes.append(JobOutcome(label=job.label, status=run.status, result=run))
                except Exception as exc:  # noqa: BLE001 - isolate, don't abort the batch
                    result.outcomes.append(
                        JobOutcome(label=job.label, status="error",
                                   error=f"{type(exc).__name__}: {exc}")
                    )
        finally:
            if owns_ledger:
                ledger.close()
        # 6 dp to match ledger_summary["est_cost_usd"] (avoid two disagreeing totals).
        result.total_cost_usd = round(ledger_cost_usd(ledger), 6)
        result.ledger_summary = conductor_summary(ledger)
        return result
