# Resource lifetime and offloading

The native runner now executes an explicitly declared sequence of computation nodes.
`ResourcePlan` scans their resource uses once, loads a resource at its first use and
releases it after its last use. Repeated forward passes are unrolled before this scan.
Resources shared by several nodes stay alive between uses. Release runs in reverse
dependency order, including after a failed allocation or computation.

This is a stage-level lifetime plan, not automatic extraction of a complete FX graph.
The runner supplies the ordered nodes and the resources each one uses.

## Native DiT

- `embed`: mapped checkpoint, embedding weights and compiled input package.
- `block_l`: mapped checkpoint, two pinned RAM buffers, two VRAM slots and the
  appropriate compiled block package. The existing worker and copy stream prefetch
  weights while the preceding block computes.
- `output`: compiled output package, loaded when first needed.

On the final pass, the input package and embedding weights end after `embed`.
After the final block, the prefetch worker joins, CUDA work completes, both staging
windows are released and the checkpoint mapping is closed. Cached pinned memory is
returned to the system. The output package is released after the final output.
The CUDA allocator cache is emptied at the end of the plan.

A CPU thread being ready does not mean its DMA has completed: stream events still
protect slot reuse. Release callbacks fence CUDA work before destroying packages.
Memory snapshots report allocated and reserved VRAM separately from Windows private
commit and resident working set. CUDA/context/library overhead can remain after all
model resources are released.

Prefetch defaults to two slots and is configurable through `--window`. An offline
profiled cost-model experiment compares windows 1–4; it selected two for the
measured configuration. No automatic runtime selection is connected yet. See
[prefetch verification](prefetch.md) for before/after scope and timing results.
`host_waited_for_weights` measures host waiting, not GPU stream stalls.

## Full causal generation

Native T5 and VAE packages do not exist yet. `generate_causal.py` therefore still
executes these stages through Python/PyTorch:

1. T5 closes its offloader even if encoding fails.
2. A function owns the entire denoiser state, including Stream, Pass, caches and
   compiled bound methods. These owners end before the full Wan VAE is constructed.
3. TAEHV is constructed at the first decoded chunk. Its weights and recurrent state
   are retained between chunks and released after the final chunk, or on failure.
4. Offloader.close() joins its worker, waits for CUDA and drops slots, queues and
   pinned buffers. stop() remains restartable for existing callers.

Python phase reports include end-of-phase allocated/reserved VRAM and pinned RAM.
Dropping an owner returns storage to PyTorch; cached pinned RAM in the Python path
may remain reserved for reuse. The native runner explicitly empties that host cache.

CPU lifecycle tests use small fake models: they check that model owners disappear
and cleanup runs on failure, not that a full Wan generation fits a memory budget.
`tests/resource_plan_test.cpp` checks stage ordering, repeated uses, dependency-order
cleanup, invalid plans and failed acquisition without CUDA.

## Verification, 2026-10-18

- Native runner compiled with MSVC and the installed PyTorch 2.10 CUDA libraries.
- The standalone C++ resource-plan checks passed.
- Python lifetime and KV-cache checks: 19 passed. Ruff passed on the changed runtime
  Python files and the new tests.
- Two native DiT forwards used the existing grid `(9, 30, 52)`, corresponding to
  480x832 spatial resolution. Both matched the saved eager reference within the
  existing tolerance: relative L2 0.0078418981, maximum absolute error 0.0234375.
  This is the same error as in the previous native report, not exact eager equality.
- Allocated VRAM was 0.624862 GiB during the block sequence and 0.00891113 GiB
  after resource release. Reserved VRAM fell from 1.60547 to 0.0449219 GiB after
  emptying the allocator cache. Windows private commit was 3.27023 GiB during
  blocks and 1.41592 GiB after cleanup. These are lifecycle snapshots, not a
  controlled before/after performance comparison or a full generation peak.
- Raw evidence: `artifacts/resource_plan_native.log`,
  `artifacts/resource_plan_check.json`, `artifacts/resource_plan_python_tests.log`.

Full T5 -> causal DiT -> VAE generation has not been rerun with this change.

## Сравнение старого и нового runner

Два прохода DiT, 480x832. GiB. Снимки после работы сняты до выхода процесса.

| Метрика | Было | Стало |
|---|---:|---:|
| RAM процесса после работы (working set) | 5.550 | 0.665 |
| Пик RAM процесса (working set) | 5.756 | 6.454 |
| Свободная RAM системы после работы | 0.642 | 6.104 |
| VRAM тензоров после работы | 0.587 | 0.009 |
| Пик VRAM тензоров | 1.351 | 1.310 |
| Резерв VRAM после работы | 1.625 | 0.045 |
| Свободная VRAM устройства после работы | 5.276 | 6.856 |

Пик физической RAM не снизился. Working set включает страницы mmap и зависит от paging Windows. Свободная RAM относится ко всей системе; это последовательные одиночные прогоны, не изолированная оценка эффекта. Native T5/VAE и полный цикл генерации в этот замер не входят.

Источник: `artifacts/resource_plan_memory_comparison.json`.
