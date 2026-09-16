# OFFLINE SIMULATION — no real provider timings

Collection only; synthesis and history excluded.
Cache reads/writes bypassed. One shared client per batch; question order is identical.
Setup and teardown are measured separately from collection in both modes.
Only pairs with all sources successful and matching result counts enter the speedup.
Matching counts do not guarantee identical live source content.

Options: `{"repeats":3,"concurrency":3,"source_timeout_seconds":10.0,"queue_timeout_seconds":10.0,"max_results_per_source":3,"warmup":false}`

| Repeat | Mode | Setup ms | Collection ms | Teardown ms |
|---|---|---:|---:|---:|
| 1 | sequential | 0.167 | 939.142 | 0.010 |
| 1 | parallel | 0.131 | 467.556 | 0.013 |
| 2 | parallel | 0.125 | 467.893 | 0.005 |
| 2 | sequential | 0.087 | 947.595 | 0.005 |
| 3 | sequential | 0.119 | 942.253 | 0.012 |
| 3 | parallel | 0.190 | 467.588 | 0.006 |

Comparable repeats: [1, 2, 3]
Median sequential ms: 942.2526000003018
Median parallel ms: 467.5884000002952
Speedup: 2.015x

Per-source observations (fetch includes the service's retries and pacing):
- wikipedia: median successful parallel fetch 46.340 ms
- arxiv: median successful parallel fetch 93.057 ms
- web: median successful parallel fetch 46.226 ms

CSV retains every measured run. Batch timing columns repeat on each source row; do not sum those columns.
There is no guaranteed speedup threshold. Offline results do not establish live provider performance.

UTC: 2026-09-16T07:38:57.223470+00:00

Runtime: Python 3.14.0, Windows

Dataset SHA256: f164885b4bcb564f4d473b495540718485e91f0d7ed562461ddd7dd77e443d2f

Git: e0e6d970e18ad595c97e53d408755dc39ea12d31; dirty=True

Service factory: scripts.c_offline:OfflineService

Reproduce: `"C:\Users\Sanan\Desktop\AI Academy\M201FinalSWE\.venv\Scripts\python.exe" scripts/benchmark.py --offline --repeats 3 --out artefacts/benchmark-offline-v2.md --csv artefacts/benchmark-offline-v2.csv`
