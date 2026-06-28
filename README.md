# 🎛️ Conductor

**A vendor-neutral, self-hosted control plane for LLM agents.**

One self-built tool-use loop drives *any* provider — Claude (official Anthropic
SDK), any OpenAI-compatible API (OpenAI / OpenRouter / Groq / Mistral / Together
/ Fireworks), or a local model (Ollama / LM Studio / vLLM) — behind a single
neutral interface, with **JSONL tracing** and a **per-provider cost ledger**.

> **Status: v2.** v0 (provider-agnostic loop + tracing + per-provider ledger) and
> v1 (sandbox isolation with snapshot/rollback, deterministic replay, local→remote
> cost cascade) are joined by v2: **multi-agent coordination under one shared cost
> budget**. Everything is verified offline (unit tests + the `demo` / `sandbox-demo`
> / `replay` / `multi-demo` commands), **and the OS isolation has been run live on a
> real Proxmox VE 9.1 LXC** (`proxmox-check` → snapshot → destructive command →
> rollback restored the marker, host untouched; the throwaway CT was destroyed
> after). So the headline "OS isolation × cost ledger × deterministic replay" is
> proven, not just claimed.

---

## Honest notes first

This project is deliberately scoped, and these limits are stated up front rather
than buried:

- **Not a new framework.** Conductor is a *self-built orchestrator on top of each
  provider's raw tool-use primitive* (Claude `tool_use` / OpenAI function
  calling). It does not wrap or invent an agent framework.
- **Vendor-neutral, and that claim is grounded, not aspirational.** The provider
  abstraction is a direct extension of [token-router](https://github.com/akagifreeez/token-router)'s
  already-working `Model`/`Usage` interface (Ollama + Fireworks + a scripted
  double, interchangeable, with a per-backend cost split). Conductor reuses that
  `Usage` and its JSONL `Ledger` verbatim and adds tool-use on top. Claude is
  **one backend among many**, reached via the *official* Anthropic SDK — not an
  OpenAI-compatible shim.
- **Two adapters cover most of the market.** `AnthropicAdapter` for Claude, and
  one `OpenAICompatAdapter` for every OpenAI-compatible endpoint (incl. local).
- **The differentiator is the *combination*, not orchestration alone.**
  Orchestrators are a crowded space; the moat is **OS isolation × cost ledger ×
  deterministic replay** together. As of v1 all three exist (the OS-isolation
  backend is verified by you on a real node — see below).
- **"Deterministic replay" = reproducing recorded I/O**, not re-running the LLM
  bit-for-bit. LLM outputs are not deterministic; the *trace* is. `conductor
  replay <trace>` re-drives the loop from a trace and reproduces its tool I/O and
  final answer (no provider calls, no tool side effects).
- **Dangerous tools are gated, and now executed in a sandbox with rollback.** The
  registry refuses a `dangerous` tool unless a sandbox is wired in. Two sandbox
  backends, at different honesty tiers:
  - `ProxmoxSandbox` — **real OS-level isolation** in a Proxmox LXC (snapshot /
    `pct exec` / rollback, mirroring the author's
    [proxmoxbot](https://github.com/akagifreeez/proxmoxbot)). This is the real
    differentiator; it needs a live Proxmox node and is **not** run by the
    offline test suite — you verify it on your homelab.
  - `SubprocessSandbox` — an **offline** double (temp-dir + copy-tree snapshot /
    rollback). It really executes and really reverts, so it proves the
    gate/snapshot/rollback *contract* without a node — but it is **NOT a security
    boundary** (a command can still touch absolute paths). It is for tests and
    the demo, not for containing hostile code.
- **AI = existing provider APIs only.** No model is trained or fine-tuned here.
  No claim that "the AI learns/predicts." Pricing figures for Claude are
  Anthropic's published per-token rates; OpenAI-family figures are approximate.
  Any model **not** in the small price table (e.g. Groq / OpenRouter / Mistral /
  Together ids) is charged a fixed placeholder rate — non-zero so cost is never
  silently $0, but it may over- or under-state the real price, so confirm /
  `Pricing.override(...)` before quoting. The ledger total is labeled
  `est_cost_usd` for this reason.
- **Near-$0 to run.** BYOK (your own provider key) or a free local model; no new
  hardware. The offline demo needs neither a key nor a network.

---

## What works today (v0)

```
user task
  └─▶ Orchestrator (self-built loop, provider-agnostic)
        ├─▶ backend.step()          ← Anthropic | OpenAI-compat | local | scripted
        │     └─ one assistant turn (text and/or tool calls), with Usage
        ├─▶ ToolRegistry.execute()  ← define-once tools; dangerous tools gated
        ├─▶ Tracer  → traces/run-<id>.jsonl   (every LLM call + tool call)
        └─▶ Ledger  → cost split keyed by provider (Usage.backend)
```

- **One loop, every provider.** `conductor run --provider {anthropic,openai-compat,local,scripted}`
  runs the *same* loop, tools, tracer, and ledger regardless of provider.
- **Define-once tools.** A tool is declared one time with a JSON-Schema spec;
  each adapter converts that into the provider's tool format.
- **Full trace.** Every LLM request/response and tool call/result is streamed to
  a JSONL file as it happens (crash-durable).
- **Per-provider cost ledger.** Reused from token-router; cost is split by
  `Usage.backend`, so a run shows what each provider cost.

## The v0 goal, reproducible offline (no key, no network)

```bash
conductor demo
```

Runs the **same task through two different provider labels** (using the offline
scripted backend) and prints:

- a **trace file per run** containing all I/O, and
- a **single cost ledger split per provider** (e.g. `anthropic` priced, `local`
  at $0).

That is the v0 acceptance criterion: *one task, two providers, all I/O traced,
cost split per provider.*

## What's new in v1

### Sandboxed dangerous tools, with rollback

```bash
conductor sandbox-demo
```

Runs an agent that lists files, executes a **destructive command**, then **rolls
back** from a pre-command snapshot — printing the box contents at each step:

```
files                     before: ['important.txt']
files  after destructive command: []          ← contained inside the sandbox
files             after rollback: ['important.txt']   ← restored
```

The command runs against the sandbox box, never the host cwd, and is fully
revertible. Offline this uses `SubprocessSandbox` (filesystem snapshot, *not* a
security boundary). For real OS isolation, point it at your Proxmox node:

```python
from conductor import Orchestrator, ToolRegistry, SANDBOX_TOOLS
from conductor.sandbox import ProxmoxSandbox

reg = ToolRegistry()
for t in SANDBOX_TOOLS:
    reg.register(t)
# env: PROXMOX_HOST / PROXMOX_USER / PROXMOX_TOKEN_NAME / PROXMOX_TOKEN_VALUE
#      PROXMOX_NODE / PROXMOX_SSH_HOST
sandbox = ProxmoxSandbox(vmid=210, node="pve")        # an LXC to run commands in
Orchestrator(backend, reg, run_id="job", sandbox=sandbox).run("...")
```

`pip install 'conductor-cp[proxmox]'` (proxmoxer + paramiko) for that backend.

#### Verify it on a real Proxmox node (one command) — done ✅

`conductor proxmox-check` runs a live snapshot → destructive command → rollback
cycle inside a real LXC and prints PASS/FAIL — the homelab counterpart to
`sandbox-demo`. It uses the *same* `sandbox_selfcheck` routine the offline suite
tests against `SubprocessSandbox`, so the verification logic is already proven;
this just points it at real hardware.

**This was run live** on a Proxmox VE 9.1 node: a throwaway debian-13 LXC was
created, the cycle PASSed (marker contained after the destructive command,
restored after rollback), and the CT was destroyed afterward — existing
containers untouched.

Two ways to connect:

```bash
pip install 'conductor-cp[proxmox]'

# A) SSH-only (no API token) — the natural fit for Tailscale SSH:
conductor proxmox-check --ssh-host 100.100.1.1 --vmid 101 \
  --template-volume local:vztmpl/debian-13-standard_13.1-2_amd64.tar.zst
#   creates a throwaway CT 101, verifies, and destroys it. Omit --template-volume
#   to use an existing (disposable!) CT instead.

# B) Proxmox API token:
export PROXMOX_HOST=192.168.1.10  PROXMOX_USER=root@pam
export PROXMOX_TOKEN_NAME=conductor  PROXMOX_TOKEN_VALUE=xxxxxxxx
export PROXMOX_NODE=pve  PROXMOX_SSH_HOST=192.168.1.10
conductor proxmox-check --vmid 210
```

Observed tail on success:

```
  [PASS] setup - proxmox-lxc via ssh (real OS isolation)
  [PASS] seed marker - conductor-selfcheck-OK
  [PASS] snapshot (call ok) - token=cdr1_selfcheck
  [PASS] destructive command
  [PASS] marker gone after destroy (contained) - __ABSENT__
  [PASS] rollback (call ok) - token=cdr1_selfcheck
  [PASS] marker restored after rollback - conductor-selfcheck-OK

PASS: sandbox snapshot/rollback self-check
```

> ⚠️ `proxmox-check` snapshots and **rolls back** the target CT — only point it at a
> disposable container (or let `--template-volume` create a throwaway one). The SSH
> sandbox only ever destroys a CT it created itself.

### Deterministic replay

```bash
conductor replay --trace traces/run-<id>.jsonl
```

Re-drives the loop from a recorded trace, returning the **recorded** tool outputs
(not re-executing) and the **recorded** assistant turns (no provider call), then
reports whether the tool I/O and final answer were reproduced. Proves a trace is
complete enough to reconstruct a run — including replaying a *sandbox* run with
no sandbox at all.

### Local→remote cost cascade

`CascadeBackend(cheap, strong)` is itself a backend: each step it tries the cheap
model (e.g. local Ollama) first and only escalates to the strong model (e.g.
Claude) when a confidence gate fails — recording **both** attempts in the ledger
so the per-provider split shows exactly what went local vs. remote. The default
gate is a small heuristic (made progress / non-hedging answer?), not a learned
router; pass your own.

## What's new in v2 — many agents, one budget

```bash
conductor multi-demo
```

A control plane should bound the combined spend of *several* agents, not just
each one in isolation. The `Coordinator` runs a list of agent jobs against a
single shared ledger and a single global `budget_usd`; once the budget is
reached, remaining agents are **skipped before they start**:

```
   agent0: completed
   agent1: completed
   agent2: completed
   agent3: skipped (budget exhausted)
   ...
ran=3  skipped=3  total=$0.001815  budget=$0.001512
```

Honest bound (stated, not hidden): the ceiling stops *further* work, so a single
in-flight step can overshoot by its own cost — the total may exceed the budget by
at most the most-expensive single step.

```python
from conductor import Coordinator, Job
result = Coordinator(budget_usd=0.50).run_all([
    Job(label="a", backend=b1, registry=r, task="..."),
    Job(label="b", backend=b2, registry=r, task="..."),
])
# result.ran / result.skipped / result.total_cost_usd / result.ledger_summary
```

## Install

> **Honest status:** Conductor is **not on PyPI**, and `pip install
> conductor-cp` does **not** work yet — it depends on
> [token-router](https://github.com/akagifreeez/token-router), which is also not
> yet published, so there is currently no clean-machine pip path. Install both
> editable from sibling checkouts (this is the supported path today):

```bash
git clone https://github.com/akagifreeez/token-router
git clone https://github.com/akagifreeez/conductor
cd conductor
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on POSIX
pip install -e ../token-router
pip install -e . --no-deps
pip install pytest
pytest -q
```

For the Claude backend, also install the optional SDK into the same venv:

```bash
pip install -e '.[anthropic]'    # or: pip install anthropic
```

(When token-router is published to PyPI, the dependency becomes a normal version
pin and `pip install conductor-cp` will work; until then the wheel's direct git
reference also blocks a PyPI upload, so the editable install above is the path.)

## Live providers (BYOK)

```bash
# Claude (needs the anthropic extra + your key)
export ANTHROPIC_API_KEY=sk-ant-...
conductor run --provider anthropic --task "What time is it in UTC? Use a tool."

# Any OpenAI-compatible endpoint
export OPENAI_API_KEY=sk-...
conductor run --provider openai-compat --model gpt-4o-mini --task "..."

# Local Ollama (no key, priced at $0)
conductor run --provider local --model qwen2.5:3b-instruct --task "..."
```

## Roadmap

- **v0 (done):** provider-agnostic tool-use loop · Anthropic + OpenAI-compat +
  local + scripted backends · JSONL tracing · per-provider cost ledger · sandbox
  gate (declared).
- **v1 (done):** sandbox executor (`ProxmoxSandbox` real OS isolation +
  `SubprocessSandbox` offline double) so dangerous tools run isolated and roll
  back · local→remote cost cascade · deterministic replay of a recorded trace ·
  `sandbox-demo` / `replay` / `proxmox-check` commands.
- **v2 (this release):** multi-agent coordination under one shared cost budget
  (`Coordinator`, `multi-demo`); an SSH-only sandbox (`ProxmoxSSHSandbox`, no API
  token — fits Tailscale SSH); **the live `proxmox-check` run on a real Proxmox VE
  9.1 node — PASS** (the last remaining "needs a homelab" item, now done).
- **Later (one of):** microVM (KVM/Firecracker) comparison · a lightweight web
  dashboard over traces/ledger.

## Tech

Python · `requests` for the OpenAI-compatible path · official `anthropic` SDK
(optional extra) for Claude · real LXC sandbox via the Proxmox API (`proxmoxer` +
`paramiko`, `[proxmox]` extra) **or** SSH-only `pct` (`ProxmoxSSHSandbox`, no
token — works over Tailscale SSH) · [token-router](https://github.com/akagifreeez/token-router)
for the `Usage`/`Ledger` accounting · no agent framework.

---

Built by [@akagifreeez](https://github.com/akagifreeez).
