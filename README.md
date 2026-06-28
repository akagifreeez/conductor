# 🎛️ Conductor

**A vendor-neutral, self-hosted control plane for LLM agents.**

One self-built tool-use loop drives *any* provider — Claude (official Anthropic
SDK), any OpenAI-compatible API (OpenAI / OpenRouter / Groq / Mistral / Together
/ Fireworks), or a local model (Ollama / LM Studio / vLLM) — behind a single
neutral interface, with **JSONL tracing** and a **per-provider cost ledger**.

> **Status: v0.** The provider-agnostic loop, tracing, and the per-provider cost
> ledger are built and verified offline (scripted backend + unit tests). The OS
> isolation (Proxmox sandbox), the cost-cascade router, and deterministic replay
> are planned for v1 — see [Roadmap](#roadmap). This README states what works
> today and what does not.

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
  Orchestrators are a crowded space; the intended moat is **OS isolation × cost
  ledger × deterministic replay** together. In v0 only the ledger + tracing
  exist; the isolation and replay are v1 (and clearly marked as such).
- **"Deterministic replay" (v1) = reproducing recorded I/O**, not re-running the
  LLM bit-for-bit. LLM outputs are not deterministic; the *trace* is.
- **Dangerous tools are gated from day one, executed in isolation in v1.** The
  registry refuses to run a tool marked `dangerous` unless a sandbox executor is
  wired in. In v0 there is no sandbox, so such tools are *declared but blocked*.
  v1 routes them into a real Proxmox LXC. The gate is real now; the isolation is
  demonstrated in v1 (not merely claimed).
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

- **v0 (now):** provider-agnostic tool-use loop · Anthropic + OpenAI-compat +
  local + scripted backends · JSONL tracing · per-provider cost ledger · sandbox
  gate (declared, not yet executing).
- **v1:** Proxmox sandbox executor (create → start → exec → snapshot → rollback)
  so dangerous tools run in a real LXC and roll back the host unharmed ·
  local↔remote cost cascade via token-router's router · deterministic replay of
  a recorded trace · 1-minute demo.
- **v2 (one of):** multi-agent coordination with a shared budget · microVM
  (KVM/Firecracker) comparison · a lightweight web dashboard.

## Tech

Python · `requests` for the OpenAI-compatible path · official `anthropic` SDK
(optional extra) for Claude · [token-router](https://github.com/akagifreeez/token-router)
for the `Usage`/`Ledger` accounting · no agent framework.

---

Built by [@akagifreeez](https://github.com/akagifreeez).
