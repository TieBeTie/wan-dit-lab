<span class="wan-scene-style wan-scene-math wan-scene-grid wan-offloading"></span>

# Offloading · MVP2

Consider the placement of resources in MVP2. The resources, the shared cache and the graph are as in [MVP1](../mvp1/Offloading.md):  $\mathcal O_{\rm offload}^{(2)}=\mathcal O_{\rm offload}^{(1)}$, except for the lines below.

$$
\underbrace{\mathrm{policy}_1=\mathrm{greedy\_ram}}_{\text{MVP1}}\mapsto\underbrace{\mathrm{policy}_2=\mathrm{greedy\_vram}}_{\text{MVP2}},\qquad\underbrace{B_{\rm VRAM}^{\rm margin}=0\ \mathrm{GiB}}_{\text{MVP1: }1.5\ \mathrm{GiB}}.\n$$

Now consider what MVP2 keeps from MVP1 and what it changes.

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Not applied</span>: the 2×2 weight windows<br>[BlockOffloader](../../../src/wan_dit_lab/offloading/offload.py) | $`\displaystyle\underbrace{2\text{ blocks in pinned RAM}}_{\text{preparation}}\longrightarrow\underbrace{2\text{ blocks in VRAM}}_{\text{computation}}`$ | The original implementation for weights; fixed prefetch depth. In MVP2 the plan comes from the calibration. | [Offloading, MVP1](../mvp1/Offloading.md) |
| **Shared resource cache**<br>[ResourceCache](../../../src/wan_dit_lab/offloading/resource_cache.py)<br><span class="wan-kept">Applied</span> | $`\displaystyle\begin{aligned}&\mathrm{VRAM}\rightleftarrows\mathrm{RAM}_{\rm cache}\rightleftarrows\mathrm{SSD}\\&\mathrm{active}(r)\Rightarrow\text{do not evict}\\&\mathrm{dirty}(r)\land\mathrm{evict}(r)\Rightarrow\mathrm{write}(r)\end{aligned}`$ | Unified budgets. First we evict the resource whose next use is farthest away; otherwise by recency of access. A clean file is reused. | |
| **Cost model**<br>[FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4)<br><span class="wan-kept">Applied</span>: calibration only | First we measure:<br>$`\displaystyle\underbrace{\Delta t_{\rm prepare}(r),\ \Delta t_{\rm transfer}(r)}_{\text{preparation and transfer of the resource}},\quad\underbrace{\Delta t_{\rm compute}(v_i)}_{\text{node execution time}}`$<br>Then we choose:<br>$`\displaystyle\underbrace{\mathrm{plan}}_{\text{what, where and when to move}},\qquad\min_{\mathrm{plan}}\widehat t_{\rm end}(\mathrm{plan})`$<br>under the RAM / VRAM budgets and the dependencies $`G`$. | A schedule based on measurements. Our profiles: medians; margin = maximum of loading and minimum of compute. Previously only the weight windows were tuned. | |
| <span class="wan-kept">Applied</span><br>policy $`\mathrm{greedy\_ram}\mapsto\mathrm{greedy\_vram}`$<br>[Cache policy](../../managed-resource-cache.md#политика-размещения-offloadingpolicy-2026-11-08) | $`\displaystyle\begin{aligned}&\text{Consider the placement of the video KV cache and of the weight rows.}\\&\underbrace{\mathrm{greedy\_ram}}_{\text{was}}:\ \min M_{\rm VRAM}\ \text{at the same speed;}\\&\qquad\text{KV and rows go to RAM or disk when their copy hides behind the block time}\\&\underbrace{\mathrm{greedy\_vram}}_{\text{now}}:\ \text{fill VRAM first:}\\&\qquad\mathrm{KV}\to\text{the GPU},\ \text{then weight rows for good,}\\&\qquad\text{the other rows from pinned RAM, the disk last}\end{aligned}`$ | Why: a placement calibrated while the card was busy left KV in RAM and cost speed. The GPU takes the KV cache first. A new policy has its own calibration key. | $`\displaystyle\begin{aligned}&B_{\rm VRAM}^{\rm margin}:\ 1.5\to0\ \mathrm{GiB}\\&\underbrace{9.6\to11.2\ \mathrm{FPS}}_{\substack{576\times1056,\ \text{NTK 1.10},\ \text{12 chunks}\\\text{KV in RAM}\to\text{KV in VRAM, margin 1 GiB}}}\end{aligned}`$<br>[Measurement 08.11](../../managed-resource-cache.md#политика-размещения-offloadingpolicy-2026-11-08) |

---
