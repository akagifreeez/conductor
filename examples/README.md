# Example artifacts

These are **real outputs** committed so you can see what Conductor produces
without cloning and running it. All were generated key-free by the offline demos
(`conductor demo` and `conductor sandbox-demo`).

| File | What it is |
|---|---|
| [`example-trace.jsonl`](example-trace.jsonl) | A run trace: every LLM request/response and tool call/result, one JSON line each (`run_start` → `llm_request` → `llm_response` → `tool_call` → `tool_result` → … → `run_end`). This is the substrate for observability and deterministic replay. |
| [`example-ledger.jsonl`](example-ledger.jsonl) | The per-provider cost ledger for a two-provider `demo` run — one row per LLM call, keyed by `backend`, so cost splits per provider. |
| [`example-sandbox-trace.jsonl`](example-sandbox-trace.jsonl) | A `sandbox-demo` trace showing the OS-isolation story: `sandbox` setup → a destructive `run_shell` (snapshotted first) → `sandbox_rollback` → teardown, with the host untouched. |

Regenerate them yourself (no API key needed):

```bash
conductor demo --trace-dir examples
conductor sandbox-demo --trace-dir examples
```

Replay a trace deterministically (reproduces the recorded tool I/O + final answer
+ terminal status, with no provider calls and no tool side effects):

```bash
conductor replay --trace examples/example-trace.jsonl
```
