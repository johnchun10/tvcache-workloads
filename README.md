# tvcache-workloads

Workload integrations for TVCache experiments. This repo is separate from
`tvclient`, `tvcache-rust`, and `container-manager-public`. It depends on them
and doesn't copy any of them.

| Workload | Owner | Status |
|---|---|---|
| [Bash / Terminal-Bench](bash/README.md) | Johnny | Working end to end on `hello-world` with real containers |
| SQL | Sameer | not in this repo yet |

```text
src/tvcache_workloads/bash/   Python package (BashToolCall, TerminalBenchSandbox, runner)
bash/                         traces, example, scripts, design docs
tests/bash/                   unit (mocked) · cache_integration (real cache) · e2e (all real)
```

Quick start: see [bash/README.md](bash/README.md).
