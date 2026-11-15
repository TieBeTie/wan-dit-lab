<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# MVP1

[README](../../README.md)

<!-- group model-timeline-frame -->
## Model

<!-- card pipeline-memory -->
## [Offloading](../../docs/research/mvp1/Offloading.md)

<!-- card model-notation-text -->
Consider video generation from a prompt. We split the latent video into chunks.

$$
\underbrace{\mathrm{ctx}}_{\text{prompt vectors}},\quad
\underbrace{\mathrm{lat}_n}_{\text{finished chunk}},\quad
\underbrace{n=0,\ldots,N-1}_{\text{chunk index}},\quad
\underbrace{\epsilon_{n+1}}_{\text{new noise}}.
$$
$$
\underbrace{W_m}_{\text{weights of the chosen model}},\quad
\underbrace{\Phi_m}_{\text{its generation rule}},\quad
m\in\{\mathrm{Wan},\underbrace{\mathrm{SF}}_{\mathrm{Self\mbox{-}Forcing}},\underbrace{\mathrm{LL}}_{\mathrm{LongLive}}\}.
$$
$$
\underbrace{\mathrm{KVvid}_n=(K_n,V_n)}_{\text{memory of finished chunks}},\quad
\underbrace{\oplus}_{\text{concatenation of K and V rows}},\quad
\underbrace{F_{\rm chunk},F_{\rm anch},F_{\rm win}}_{\text{latent frames: chunk, anchor, window}}.
$$

Now consider what the model generates in one call and on what context it was trained.

<!-- card long-generation -->
## [Long-generation](../../docs/research/mvp1/Long-generation.md)

<!-- card scene-blackboard -->
## [Scene-change](../../docs/research/mvp1/Scene-change.md)

<!-- card runtime-optimization -->
## [Optimization](../../docs/research/mvp1/Optimization.md)

<!-- card rope-positions -->
## [RoPE-positions](../../docs/research/mvp1/RoPE-positions.md)

<!-- card compilation -->
# Compilation · MVP1

Consider the computation graph of one DiT block.

$$
\underbrace{G=(V,E)}_{\text{computation graph}},\qquad
\underbrace{V}_{\text{operations}},\quad
\underbrace{E}_{\text{data dependencies}}.
$$

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| $\displaystyle \mathrm{SDPA}\mapsto\mathrm{Sage\\,2}$ | $\displaystyle \begin{aligned}&\text{Consider the self- and cross-attention of the block.}\\&\mathrm{Attn}\_{\rm self}=\mathrm{Attn}\_{\rm cross}=\mathrm{Sage\\,2}\\&\mathrm{Linear}=\mathrm{Nunchaku\ W4A4}\end{aligned}$ | We choose the kernels of the graph operations. | $\displaystyle \begin{aligned}&\underbrace{T\_{\rm block}:186.8\to77.3\ \mathrm{ms}}\_{\text{only self-attention replaced}}\\&\underbrace{704\times1280,\quad\mathrm{FP8}}\_{\text{comparison conditions E29}}\end{aligned}$<br>E29 |
| $\displaystyle (W\_Q,W\_K,W\_V)\mapsto W\_{\rm QKV}$ | $\displaystyle \begin{aligned}&\text{Consider the three self-attention projections of the input }X.\\&W\_{\rm QKV}^{\top}=[W\_Q^{\top}\ W\_K^{\top}\ W\_V^{\top}]\\&b\_{\rm QKV}=[b\_Q\ b\_K\ b\_V]\\&(Q,K,V)=\mathrm{split}(XW\_{\rm QKV}^{\top}+b\_{\rm QKV})\end{aligned}$ | One main GEMM instead of three; the input is quantized once. | $\displaystyle \begin{aligned}&T\_{\rm QKV}:4.32\to3.77\ \mathrm{ms}\\&\underbrace{\text{block 0, 7040 tokens}}\_{\text{separate measurement E24}}\end{aligned}$<br>[E24](../../notes/research/cards/E24.%20%D0%A1%D0%BB%D0%B8%D1%82%D1%8B%D0%B5%20QKV%20%D0%B8%20FFN%20%D0%B2%20W4A4,%20RoPE%20%D0%B2%D0%BE%20float32,%20704%C3%971280.md) · [Implementation](../../src/wan_dit_lab/quantization/w4a4.py) |
| $\displaystyle \mathrm{FFN}\mapsto\mathrm{FFN}\_{\rm fused}$ | $\displaystyle \begin{aligned}&\text{Consider the two linear layers of the FFN.}\\&\underbrace{\mathrm{GEMM}\_1\to\mathrm{GELU}\to\mathrm{Quant}\_4}\_{\text{first GEMM with a fused epilogue}}\\&\qquad\to\mathrm{GEMM}\_2\end{aligned}$ | In the epilogue we also compute the projection of the low-rank branch of the second layer; the intermediate hidden is not written in 16 bits. | $\displaystyle \begin{aligned}&T\_{\rm FFN}:12.70\to10.67\ \mathrm{ms}\\&\underbrace{\text{block 0, 7040 tokens}}\_{\text{separate measurement E24}}\end{aligned}$<br>[E24](../../notes/research/cards/E24.%20%D0%A1%D0%BB%D0%B8%D1%82%D1%8B%D0%B5%20QKV%20%D0%B8%20FFN%20%D0%B2%20W4A4,%20RoPE%20%D0%B2%D0%BE%20float32,%20704%C3%971280.md) · Implementation |
| $\displaystyle \mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}\_{\rm Sage}$ | $\displaystyle \begin{aligned}&\text{Consider the preparation of }Q,K\text{ for Sage.}\\&\mathrm{RMSNorm}\to\mathrm{RoPE}\to\mathrm{Quant}\_{\rm INT8}^{Q,K}\\&\qquad\to\mathrm{Attn}\_{\rm Sage}\end{aligned}$ | We rotate Q/K while preparing the quantized data for Sage. | The RMS statistics are computed in a separate pass.<br>Implementation |
| $\displaystyle G\xrightarrow{\mathcal O}G'\xrightarrow{\mathcal C\_{\rm compile}}\mathrm{package}$ | $\displaystyle \begin{aligned}&\text{Consider the graph after the choice of kernels and fusions.}\\&G'\xrightarrow{\rm torch.export}G\_{\rm FX}\\&G\_{\rm FX}\xrightarrow{\rm AOTInductor}\mathrm{package}\end{aligned}$ | We build a package for running; QKV, FFN and RoPE are enabled independently. | [Live 1](../../configs/live.json) · [Export](../../scripts/conversion/export_live.py) |

<!-- card model-history -->
## [Model](../../docs/research/mvp1/Model.md)

<!-- card model-text-decoder -->
## [Model](../../docs/research/mvp1/Model.md)

<!-- card longlive-boundary -->
---

**LongLive choice · next stage**

<!-- card model-after-longlive -->
## [Model](../../docs/research/mvp1/Model.md)
