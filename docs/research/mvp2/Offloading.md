<span class="wan-scene-style wan-scene-math wan-scene-grid wan-offloading"></span>

# Offloading

Consider the generation of the next chunk. The rectangles hold the necessary data and weights.

$$
\underbrace{\boxed{\mathrm{prompt}}}_{\text{prompt}}
\xrightarrow[\boxed{W^{\rm text}}]{\text{T5}}
\underbrace{\boxed{\mathrm{ctx}}}_{\text{text condition}}
$$
$$
\left(\underbrace{\boxed{\epsilon_{n+1}}}_{\text{noise}},\boxed{\mathrm{ctx}},
\underbrace{\boxed{\mathrm{KVvid}_n}}_{\text{video KV}}\right)
\xrightarrow[\boxed{W^{\rm DiT}}]{\text{DiT}}
\left(\underbrace{\boxed{\mathrm{KVvid}_{n+1}}}_{\text{updated self-attention KV}},
\underbrace{\boxed{\mathrm{lat}_{n+1}}}_{\text{latent chunk}}\right)
$$
$$
\boxed{\mathrm{lat}_{n+1}}
\xrightarrow[\boxed{W^{\rm dec}}]{\text{VAE decoder}}
\underbrace{\boxed{\mathrm{frame}_{n+1}}}_{\text{frames}}
\longrightarrow\mathrm{MP4}
$$
$$
\underbrace{\boxed{\mathrm{ctx}},\ \boxed{\mathrm{KVvid}_{n+1}}}_{\text{kept between chunks}},\qquad
\underbrace{\boxed{\mathrm{KVtext}(\mathrm{ctx})},\ \boxed{\mathrm{lat}_{\rm win}},\ \boxed{\mathrm{state}_{\rm dec}}}_{\substack{\text{by mode: text KV, latents for recache,}\\\text{state of the streaming decoder}}}.
$$

Let us combine this with the working buffers and the computation code into resources.

$$
\begin{aligned}
\underbrace{\mathcal R}_{\text{resources}}
={}&\underbrace{\mathcal R_W}_{\text{weights}}
\cup\underbrace{\mathcal R_{\rm state}}_{\text{KV, decoder state}}
\cup\underbrace{\mathcal R_x}_{\text{activations, working buffers}}\\
&\cup\underbrace{\mathcal R_{\rm data}}_{\text{inputs, conditions, latents, frames}}
\cup\underbrace{\mathcal R_{\rm code}}_{\text{computation packages, libraries}}.
\end{aligned}
$$

We place the kept tensors in a common cache:

$$
\mathrm{VRAM}\rightleftarrows
\underbrace{\mathrm{RAM}_{\rm cache}}_{\text{limited budget}}
\rightleftarrows\mathrm{SSD}.
$$

Consider the order of computations and the resources of each node:

$$
\underbrace{G=(V,E)}_{\text{computations and dependencies}},\qquad
\underbrace{v_1,\ldots,v_N}_{\text{execution order}},\qquad
\underbrace{\mathcal R_i\subseteq\mathcal R}_{\text{resources of node }v_i}.
$$

| <span class="wan-offload-methods">Method / source</span> | Context → computation | Idea |
| :--- | :--- | :--- |
| **Lifetime graph**<br>[ResourcePlan](../../src/wan_dit_lab/offloading/resource_schedule.py)<br>Related approaches: [SwapAdvisor](https://par.nsf.gov/servlets/purl/10191573), [TensorShunt](https://github.com/eladwf/TensorShunt/blob/main/python/tensorshunt/schedule.py) | $\displaystyle\begin{aligned}&\underbrace{\mathrm{first}(r)=\min\{i:r\in\mathcal R_i\}}_{\text{first use}}\\&\underbrace{\mathrm{last}(r)=\max\{i:r\in\mathcal R_i\}}_{\text{last use}}\\&\mathrm{acquire}(r)\to v_{\mathrm{first}(r)}\to\cdots\to v_{\mathrm{last}(r)}\to\mathrm{release}(r)\end{aligned}$ | When a resource is needed and when it can be released. Before use we wait for the load, before release we wait for CUDA to finish. |
| **Weight windows 2×2**<br>[BlockOffloader](../../src/wan_dit_lab/offloading/offload.py)<br>[Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) | $\displaystyle\underbrace{2\text{ blocks in pinned RAM}}_{\text{preparation}}\longrightarrow\underbrace{2\text{ blocks in VRAM}}_{\text{computation}}$ | The original implementation for weights. Fixed prefetch depth. |
| **Common resource cache**<br>[ResourceCache](../../src/wan_dit_lab/offloading/resource_cache.py)<br>**Applied** | $\displaystyle\begin{aligned}&\mathrm{VRAM}\rightleftarrows\mathrm{RAM}_{\rm cache}\rightleftarrows\mathrm{SSD}\\&\mathrm{active}(r)\Rightarrow\text{we do not evict}\\&\mathrm{dirty}(r)\land\mathrm{evict}(r)\Rightarrow\mathrm{write}(r)\end{aligned}$ | Unified budgets. First we evict the resource with the farthest next use; otherwise by recency of access. A clean file is reused. |
| **Cost model**<br>[FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4)<br>**Connected to the common cache** | First we measure:<br>$\displaystyle\underbrace{\Delta t_{\rm prepare}(r),\ \Delta t_{\rm transfer}(r)}_{\text{preparation and transfer of the resource}},\quad\underbrace{\Delta t_{\rm compute}(v_i)}_{\text{execution time of the node}}$<br>Then we choose:<br>$\displaystyle\underbrace{\mathrm{plan}}_{\text{what, where and when to transfer}},\qquad\min_{\mathrm{plan}}\widehat t_{\rm end}(\mathrm{plan})$<br>under the RAM / VRAM budgets and the dependencies $G$. | A schedule from measurements. Our profiles: medians; margin = the maximum of the load and the minimum of the computation. Earlier only the weight windows were tuned. |

$$
\Delta t_{\rm disk}(r)=
\frac{\underbrace{B_{\rm read}(r)}_{\text{bytes read}}}{\underbrace{\beta_{\rm read}}_{\text{bytes/s}}}
+
\frac{\underbrace{B_{\rm write}(r)}_{\text{bytes written}}}{\underbrace{\beta_{\rm write}}_{\text{bytes/s}}}.
$$

[Measurements and conditions of the previous runs](cards/H12.%20Подгрузка%20блоков%20даст%20полноразмерный%20BF16%20эталон.md#Предыдущие%20короткие%20прогоны)

**Sources:** [FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4) · [Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) · [PyTorch: mmap](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html#using-torch-load-mmap-true).
