"""Command-line entry point.

Subcommands:

* ``conductor run`` - run one task through one provider. ``--provider`` selects
  the backend; the SAME loop, tools, tracer, and ledger run regardless of which
  provider you pick. That interchangeability is the whole point.

* ``conductor demo`` - the offline v0 goal, no network or API key required: the
  same task is run through two *different* provider labels (using the scripted
  backend) and the result shows (a) a trace file per run with all I/O and (b) a
  cost ledger split per provider.

* ``conductor sandbox-demo`` - the offline v1 sandbox proof: a destructive
  command runs inside a sandbox, is contained, and is rolled back from a
  snapshot - with the host untouched. Uses the SubprocessSandbox double (no
  Proxmox needed); the real OS isolation is ProxmoxSandbox.

* ``conductor replay`` - re-run a recorded trace deterministically, reproducing
  its tool I/O and final answer (no provider calls, no tool side effects).

* ``conductor multi-demo`` - the offline v2 proof: many agents share ONE cost
  budget; once it's spent, remaining agents are skipped (global ceiling).

* ``conductor proxmox-check`` - LIVE verification on a real Proxmox node: snapshot
  -> destructive command -> rollback inside an LXC, printing PASS/FAIL.
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
from .coordinator import Coordinator, Job
from .orchestrator import Orchestrator, conductor_summary, ledger_cost_usd
from .pricing import make_pricing
from .replay import load_trace, replay_trace
from .sandbox import SubprocessSandbox
from .tools import READONLY_TOOLS, SANDBOX_TOOLS, ToolRegistry


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


def cmd_sandbox_demo(args: argparse.Namespace) -> int:
    """Offline v1 proof: destructive command -> contained -> rolled back."""
    py = sys.executable
    # Portable file ops via the running interpreter (works on Windows + Linux).
    # chr(42)='*', chr(47)='/' to avoid nested-quote headaches in the -c string.
    _listing = "sorted(p.replace(os.sep,chr(47)) for p in glob.glob(chr(42)) if os.path.isfile(p))"
    list_cmd = f'"{py}" -c "import glob,os;print({_listing})"'
    # Destroy everything, THEN print the (now empty) listing - so the command's own
    # output proves the destruction was contained inside the box.
    destroy_cmd = (
        f'"{py}" -c "import glob,os,shutil;'
        f"[shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob(chr(42))];"
        f'print({_listing})"'
    )
    reg = ToolRegistry()
    for tool in SANDBOX_TOOLS:
        reg.register(tool)
    # No hardcoded snapshot token: sandbox_rollback with no arg reverts the most
    # recent run_shell (here, the destructive one).
    script = [
        ScriptedTurn(text="listing the sandbox", tool_calls=[("run_shell", {"command": list_cmd})]),
        ScriptedTurn(text="running a destructive command (delete-all)", tool_calls=[("run_shell", {"command": destroy_cmd})]),
        ScriptedTurn(text="rolling back the destructive command", tool_calls=[("sandbox_rollback", {})]),
        ScriptedTurn(text="listing after rollback", tool_calls=[("run_shell", {"command": list_cmd})]),
        ScriptedTurn(text="The destructive command ran inside the sandbox, was contained, and was rolled back."),
    ]
    backend = ScriptedBackend(script, name="scripted-agent", backend="scripted")
    sandbox = SubprocessSandbox(seed_files={"important.txt": "keep me", "logs/app.log": "data"})
    print("Sandbox demo: a destructive command, contained and rolled back (offline)\n")
    res = Orchestrator(
        backend, reg, run_id=_run_id("sandbox"), trace_dir=args.trace_dir,
        max_steps=10, sandbox=sandbox,
    ).run("List the files, destroy them, then roll back.")

    rows = load_trace(res.trace_path)
    listings = []
    for r in rows:
        if r["kind"] == "tool_result" and r["content"].startswith("{") and "stdout" in r["content"]:
            import json as _json
            out = _json.loads(r["content"]).get("stdout", "")
            if out.strip().startswith("["):
                listings.append(out.strip())
    labels = ["before", "after destructive command", "after rollback"]
    for label, out in zip(labels, listings):
        print(f"  files {label:>26}: {out}")
    print(f"\nstatus={res.status}  trace: {res.trace_path}")
    print(format_ledger(res.ledger_summary))
    print(
        "\nThe destructive command executed in the sandbox box dir, never the host "
        "cwd, and rollback restored the files. (Offline double = filesystem "
        "snapshot, NOT a security boundary; real OS isolation is ProxmoxSandbox.)"
    )
    return 0


def cmd_multi_demo(args: argparse.Namespace) -> int:
    """Offline v2 proof: many agents share ONE budget; later agents are cut off."""
    def make_agent(i: int) -> ScriptedBackend:
        return ScriptedBackend(
            [ScriptedTurn(text=f"Agent {i} completed its task with a reasonably detailed answer.")],
            name="claude-opus-4-8", backend="anthropic",
        )

    def make_reg() -> ToolRegistry:
        r = ToolRegistry()
        for t in READONLY_TOOLS:
            r.register(t)
        return r

    task = "Summarize the assigned task."
    # Measure one agent's cost so the demo budget is meaningful regardless of
    # token-estimate specifics, then set a budget that admits ~2-3 of N agents.
    probe = Ledger(pricing=make_pricing())
    Orchestrator(make_agent(0), make_reg(), run_id=_run_id("probe"),
                 trace_dir=args.trace_dir, ledger=probe).run(task)
    per_agent = ledger_cost_usd(probe)
    probe.close()
    budget = args.budget if args.budget is not None else round(2.5 * per_agent, 8) or 1e-4

    n = args.agents
    jobs = [Job(label=f"agent{i}", backend=make_agent(i), registry=make_reg(), task=task)
            for i in range(n)]
    coord = Coordinator(budget_usd=budget, trace_dir=args.trace_dir)
    result = coord.run_all(jobs)

    print(f"{n} agents share ONE budget of ${budget:.6f} "
          f"(~${per_agent:.6f}/agent) - offline\n")
    for o in result.outcomes:
        note = "skipped (budget exhausted)" if o.status == "skipped_budget" else o.status
        print(f"  {o.label:>8}: {note}")
    print(f"\nran={len(result.ran)}  skipped={len(result.skipped)}  "
          f"total=${result.total_cost_usd:.6f}  budget=${budget:.6f}")
    print(format_ledger(result.ledger_summary))
    print(
        "\nThe shared budget capped total agent spend: once it was reached, "
        "remaining agents were skipped before starting (a single in-flight step "
        "can overshoot by its own cost; the ceiling stops further work)."
    )
    return 0


def cmd_proxmox_check(args: argparse.Namespace) -> int:
    """Live verification on a REAL Proxmox node: snapshot/destroy/rollback an LXC.

    Run this on your homelab. It needs `conductor-cp[proxmox]` (proxmoxer +
    paramiko), the PROXMOX_* env vars, and an LXC (existing --vmid, or --template
    to clone). It uses the same sandbox_selfcheck logic the offline suite tests.
    """
    from .sandbox import ProxmoxSandbox, posix_commands, sandbox_selfcheck

    sandbox = ProxmoxSandbox(
        vmid=args.vmid,
        node=args.node,
        template_vmid=args.template,
    )
    # Default marker path is on the rootfs (/root); override if your CT differs.
    cmds = posix_commands(path=args.path) if args.path else posix_commands()
    print(f"Proxmox live self-check: vmid={args.vmid} node={args.node or '(env PROXMOX_NODE)'}\n")
    report = sandbox_selfcheck(sandbox, **cmds)
    print(report.format())
    if report.error and ("proxmoxer" in report.error or "paramiko" in report.error):
        print("\nInstall the extra:  pip install 'conductor-cp[proxmox]'")
    return 0 if report.passed else 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-run a recorded trace deterministically and report the match."""
    res, cmp = replay_trace(args.trace, run_id=_run_id("replay"), trace_dir=args.trace_dir)
    print(f"replayed {args.trace}")
    print(f"  -> new trace: {res.trace_path}")
    n_orig, n_rep = cmp["n_tool_results"], cmp["n_tool_results_replayed"]
    if n_orig == 0:
        print("  tool I/O reproduced: n/a (no tool calls in trace)")
    else:
        counts = f"{n_rep}/{n_orig}" if n_rep != n_orig else f"{n_orig}"
        print(f"  tool I/O reproduced: {cmp['tool_results_match']}  ({counts} tool result(s))")
    print(f"  final answer reproduced: {cmp['final_match']}")
    print(f"  overall match: {cmp['match']}")
    if not cmp["match"]:
        print(f"  original final: {cmp['original_final']!r}")
        print(f"  replayed final: {cmp['replayed_final']!r}")
    return 0 if cmp["match"] else 1


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

    ps = sub.add_parser(
        "sandbox-demo",
        help="offline v1 sandbox proof (destructive cmd contained + rolled back)",
    )
    ps.add_argument("--trace-dir", default="traces")
    ps.set_defaults(func=cmd_sandbox_demo)

    prp = sub.add_parser("replay", help="deterministically replay a recorded trace")
    prp.add_argument("--trace", required=True, help="path to a run-*.jsonl trace")
    prp.add_argument("--trace-dir", default="traces", help="where to write the replay trace")
    prp.set_defaults(func=cmd_replay)

    pmd = sub.add_parser(
        "multi-demo",
        help="offline v2 proof: many agents share one budget; later agents cut off",
    )
    pmd.add_argument("--agents", type=int, default=6, help="number of agents")
    pmd.add_argument("--budget", type=float, default=None, help="shared budget USD (default: ~2.5 agents)")
    pmd.add_argument("--trace-dir", default="traces")
    pmd.set_defaults(func=cmd_multi_demo)

    ppx = sub.add_parser(
        "proxmox-check",
        help="LIVE verification on a real Proxmox node (snapshot/destroy/rollback an LXC)",
    )
    ppx.add_argument("--vmid", type=int, required=True, help="LXC id to run the check in")
    ppx.add_argument("--node", default=None, help="Proxmox node (or env PROXMOX_NODE)")
    ppx.add_argument("--template", type=int, default=None,
                     help="optional template vmid to clone a fresh CT from (destroyed after)")
    ppx.add_argument("--path", default=None,
                     help="marker file path inside the CT (default /root/...; avoid tmpfs /tmp)")
    ppx.set_defaults(func=cmd_proxmox_check)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
