"""Command-line entry point.

Two subcommands:

* ``conductor run`` - run one task through one provider. ``--provider`` selects
  the backend; the SAME loop, tools, tracer, and ledger run regardless of which
  provider you pick. That interchangeability is the whole point.

* ``conductor demo`` - the offline v0 goal, no network or API key required: the
  same task is run through two *different* provider labels (using the scripted
  backend) and the result shows (a) a trace file per run with all I/O and (b) a
  cost ledger split per provider. This is the reproducible proof that the loop is
  provider-agnostic and the accounting is per-provider.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import List, Optional

from token_router.accounting import Ledger

from .backends.base import AgentBackend
from .backends.scripted import ScriptedBackend, ScriptedTurn
from .orchestrator import Orchestrator, conductor_summary
from .pricing import make_pricing
from .tools import READONLY_TOOLS, ToolRegistry


def _run_id(tag: str) -> str:
    # Millisecond clock + a short random suffix so two runs with the same tag in
    # the same second get distinct ids (the trace file is keyed by run_id and
    # opened truncating - a collision would silently overwrite the earlier run).
    return f"{int(time.time() * 1000)}-{tag}-{uuid.uuid4().hex[:6]}"


def _build_backend(provider: str, model: Optional[str], base_url: Optional[str]) -> AgentBackend:
    """Construct the backend for ``--provider``. Network adapters import lazily."""
    if provider == "anthropic":
        from .backends.anthropic_adapter import AnthropicAdapter, DEFAULT_MODEL

        return AnthropicAdapter(model or DEFAULT_MODEL)
    if provider == "openai-compat":
        from .backends.openai_compat import OpenAICompatAdapter

        if not model:
            raise SystemExit("--model is required for --provider openai-compat")
        return OpenAICompatAdapter(model, base_url=base_url, backend="openai-compat")
    if provider == "local":
        from .backends.openai_compat import OpenAICompatAdapter

        # Local Ollama/LM Studio: backend label "local" -> priced at $0.
        return OpenAICompatAdapter(
            model or "qwen2.5:3b-instruct", base_url=base_url, backend="local"
        )
    if provider == "scripted":
        return ScriptedBackend(
            _smoke_script(), name=model or "scripted-1", backend="scripted"
        )
    raise SystemExit(f"unknown provider: {provider}")


def _smoke_script() -> List[ScriptedTurn]:
    """A tiny canned trajectory: call the `now` tool, then answer."""
    return [
        ScriptedTurn(tool_calls=[("now", {})]),
        ScriptedTurn(text="Done - I called the `now` tool and reported the current UTC time."),
    ]


def format_ledger(summary: dict) -> str:
    """Conductor-flavored ledger summary, focused on the per-provider split."""
    lines = [
        "=== conductor cost ledger ===",
        f"runs(tasks)={summary['tasks']}  llm_calls={summary['calls']}  "
        f"total_tokens={summary['total_tokens']:,}  est_cost_usd=${summary['est_cost_usd']:.6f}",
        "per-provider:",
    ]
    for backend, d in summary["by_backend"].items():
        lines.append(
            f"  [{backend}] calls={d['calls']}  tokens={d['total_tokens']:,}  "
            f"cost=${d['cost_usd']:.6f}"
        )
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    if not args.task.strip():
        raise SystemExit("--task must not be empty")
    backend = _build_backend(args.provider, args.model, args.base_url)
    registry = ToolRegistry()
    for tool in READONLY_TOOLS:
        registry.register(tool)

    orch = Orchestrator(
        backend,
        registry,
        run_id=_run_id(args.provider),
        max_steps=args.max_steps,
        trace_dir=args.trace_dir,
    )
    result = orch.run(args.task)

    print(f"\n[provider={backend.backend} model={backend.name}]")
    print(f"status={result.status}  steps={result.steps}")
    print(f"final: {result.final_text}")
    print(f"trace: {result.trace_path}")
    print(format_ledger(result.ledger_summary))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the same task through two provider labels with one shared ledger."""
    task = args.task or "What is the current UTC time? Use the `now` tool, then tell me."
    registry = ToolRegistry()
    for tool in READONLY_TOOLS:
        registry.register(tool)

    # One shared ledger across both providers (so the split aggregates), streamed
    # to its own JSONL for crash-durability, and closed by us since the
    # orchestrator only closes ledgers it owns.
    ledger_path = os.path.join(args.trace_dir, f"ledger-demo-{uuid.uuid4().hex[:6]}.jsonl")
    shared = Ledger(pricing=make_pricing(), jsonl_path=ledger_path)
    providers = [
        ("anthropic", "claude-opus-4-8"),
        ("local", "qwen2.5:3b-instruct"),
    ]
    print("Running the SAME task through two providers (offline scripted backends)\n")
    try:
        for label, model in providers:
            backend = ScriptedBackend(_smoke_script(), name=model, backend=label)
            orch = Orchestrator(
                backend,
                registry,
                run_id=_run_id(label),
                max_steps=args.max_steps,
                trace_dir=args.trace_dir,
                ledger=shared,  # shared so the cost split aggregates across providers
            )
            result = orch.run(task)
            print(f"[{label}] status={result.status} steps={result.steps} -> trace: {result.trace_path}")

        print()
        print(format_ledger(conductor_summary(shared)))
        print(f"ledger: {ledger_path}")
        print(
            "\nGoal met: one task, two providers, all I/O in per-run traces, "
            "cost split per provider in a single ledger."
        )
    finally:
        shared.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="conductor", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run one task through one provider")
    pr.add_argument(
        "--provider",
        required=True,
        choices=["anthropic", "openai-compat", "local", "scripted"],
        help="which backend to drive (same loop for all)",
    )
    pr.add_argument("--task", required=True, help="the task/prompt for the agent")
    pr.add_argument("--model", default=None, help="model id (provider-specific)")
    pr.add_argument("--base-url", default=None, help="override base URL (openai-compat/local)")
    pr.add_argument("--max-steps", type=int, default=8, help="max tool-use iterations")
    pr.add_argument("--trace-dir", default="traces", help="where to write JSONL traces")
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("demo", help="offline two-provider goal (no key/network)")
    pd.add_argument("--task", default=None, help="override the demo task")
    pd.add_argument("--max-steps", type=int, default=8)
    pd.add_argument("--trace-dir", default="traces")
    pd.set_defaults(func=cmd_demo)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
