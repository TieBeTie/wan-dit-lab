# RAM budget for interactive use

Implemented native foundation, 2026-11-03. Full point-A model smoke remains pending.

`offloading.ram_margin_fraction` is a fraction of total physical RAM to
leave available to the system. Suggested initial value: 0.125, approximately
2 GiB on this machine. This is not a fraction of currently free RAM and not
an OS reservation or a guarantee against unrelated applications allocating.

Read total and available physical RAM with GlobalMemoryStatusEx. For a new
allocation, the headroom is max(0, available - total * ram_margin_fraction),
less known pending working/staging allocations. Existing allocations must
not be charged twice. Count pinned buffers explicitly; process working set
alone is not a sufficient accounting source.

Weights, KV, T5 and transfer buffers share this admission budget. RAM/VRAM/SSD
placement must obey capacity before choosing the fastest calibrated plan.
If the mandatory working buffers do not fit, stop with the measured required
and available amounts. Do not rely on the pagefile as pinned-memory capacity.

During generation, reassess at safe chunk boundaries. When system headroom
falls below the reserve, discard eligible immutable weight caches and read them
from their existing disk packs. Automatic KV placement stays in RAM/VRAM or
refuses startup; it must never create mutable SSD backing. Do not re-admit during the same run after pressure eviction. This conservative policy avoids oscillation; adaptive readmission is deferred. Changes
by unrelated processes can temporarily violate the target; a scalar setting
cannot guarantee desktop responsiveness. Monitor disk contention as well.

Only an explicitly authorized `kv_cache: "disk"` uses mutable disk backing.
For that opt-in path, persist changed committed ranges only. Use bounded pinned
staging and prefetch through the common native pipeline. Temporary backing
files have an explicit lifetime and are removed when the run closes normally.
Keep output videos, configurations and reports independently of backing files.

Existing implementation references:

- Python: `src/wan_dit_lab/offloading/resource_store.py`, `resource_cache.py`;
  `system_ram_reserve_gib`, `process_rss_limit_gib`, RAM and disk backing budgets.
- Native: `budget.rs` gates pinned allocations using current physical availability.
  `cache.rs` uses two GPU and two pinned staging slots for disk KV. `kv_disk.rs`
  owns a temporary backing file, range checks, byte counters and cleanup.
  `Rows` promotes disk weights directly to VRAM without materializing all rows in RAM.

Validation before full render: failing behavior tests, bounded RAM/SSD KV
round-trip and dirty-range tests, GPU comparison against RAM-backed KV, then
native smoke measuring steady FPS, prompt-change latency, available RAM minimum,
process working set/private commit, pinned RAM, disk bytes read/written and
transfer waits. Estimate the three selected scenes (57 chunks, 13 prompt steps)
from these measurements, including encoding overhead and startup separately.
A full generation requires the user's current go-ahead; old timing estimates
do not authorize a new experiment.

Sources:

- https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex
- https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex
- https://learn.microsoft.com/en-us/windows/apps/develop/performance/disk-memory

Implementation limits and checks:

- Old configurations default to RAM fraction 0; point A explicitly uses 0.125.
- The reserve is an admission target, not an OS-enforced reservation. Ordinary
  Rust allocations, exported model workspaces and OS file cache are not individually
  intercepted. Current availability includes their physical pressure at each probe.
- KV disk I/O currently uses synchronous file operations and Windows file caching.
  GPU transfers retain event ordering and bounded staging; CPU disk I/O can stall
  scheduling. This is a correctness baseline, not a throughput claim or an async
  disk scheduler. Disk byte counters are logical I/O, not physical SSD writes.
- Temporary backing uses the system temporary directory and delete-on-close on
  Windows. Insufficient disk space propagates as an error. It never lives among videos.
- Startup checks pending KV, weight staging, pixels and T5 staging together.
  Every native pinned allocation rechecks availability under a shared allocation
  lock. Failed admission stops with required/available memory, not pagefile fallback.
- Capacity guards include VRAM margin and measured generation workspace during
  calibration transitions. Saved placement is checked against current resources.
- CUDA integration compares RAM and SSD KV for FP8 and BF16 over multiple commits,
  slot reuse, reads back to GPU, ring-copy destinations and RAM-to-disk eviction.
  This test uses small buffers; it is not full-model performance validation.

Verified 2026-11-03: 53 Rust tests passed (8 GPU/model tests excluded from default run), 58 Python pipeline tests passed. Two bounded CUDA tests were run explicitly and passed: KV RAM/SSD equivalence and direct disk-weight promotion with RAM/disk transitions. Six full-model integration tests remain unrun. Release and debug executables rebuilt. No full generation was launched.
