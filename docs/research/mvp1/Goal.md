<span class="wan-scene-style wan-scene-math wan-scene-grid wan-version-matrix"></span>
# Research goal

## Context

$$
\begin{aligned}
\mathrm{prompt}&\xrightarrow{\mathrm{cfg}}\mathrm{video},&&\text{consider video generation from a prompt in live mode,}\\
\mathrm{cfg}&\in\mathcal C,&&\text{with configurations from the set C,}\\
\mathrm{FPS}(\mathrm{cfg})&=\frac{N_{\rm frames}}{T_{\rm generation}(\mathrm{cfg})}\ge 15,&&\text{Live mode requires a rate of at least 15 frames per second.}\\
M_{\rm VRAM}(\mathrm{cfg})&\le 8\ \text{GB},\quad M_{\rm RAM}(\mathrm{cfg})\le 16\ \text{GB},&&\text{The difficulty is that the memory of the NVIDIA GeForce RTX 4060 Laptop GPU is limited,}\\
\mathcal L(\mathrm{cfg})&=\mathrm{LPIPS}_{\rm align}(\mathrm{cfg})+\bigl(1-\mathrm{DINO}(\mathrm{cfg})\bigr).&&\text{and speeding up video generation lowers the video quality, which we measure by the loss.}
\end{aligned}
$$

## Goal

Find the configuration that achieves the smallest video quality loss<br>at a rate of at least 15 frames per second<br>on the NVIDIA GeForce RTX 4060 Laptop (8 GB of video memory, 16 GB of RAM).

Let us state the optimization problem

$$
\begin{aligned}
\min_{\mathrm{cfg}\in\mathcal C}\quad&\mathcal L(\mathrm{cfg})\\
\text{subject to}\quad&\mathrm{FPS}(\mathrm{cfg})\ge 15,\\
&M_{\rm VRAM}(\mathrm{cfg})\le 8\ \text{GB},\\
&M_{\rm RAM}(\mathrm{cfg})\le 16\ \text{GB}.
\end{aligned}
$$

# Task 1

Build a mathematical model that predicts $\mathrm{FPS}$, memory and $\mathcal L$ from the configuration $\mathrm{cfg}$, so that configurations violating the constraints can be discarded even before experiments.

$$
\mathrm{cfg}=\Bigl(\underbrace{\mathrm{cfg}_{\rm offloading}}_{\text{system loading plan}},\quad \underbrace{\mathrm{cfg}_{\rm input}}_{\text{input}},\quad \underbrace{\mathrm{cfg}_{\rm text\ encoder},\ \underbrace{\mathrm{cfg}_{\rm DiT}}_{\text{transformer block configuration}},\ \mathrm{cfg}_{\rm VAE}}_{\text{video generation}}\Bigr)
$$



| Version | Model | Quantization | Compilation | Resolution change | Offloading | Video continuation | Scene change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| General model | $\displaystyle \begin{aligned} \mathrm{prompt}\_j&\xrightarrow{\mathrm{T5}}\mathrm{ctx}\_j,\qquad \mathrm{KVtext}\_j=\mathrm{KV}^{\rm cross}(\mathrm{ctx}\_j),\\\\ (\epsilon\_{n+1},\mathrm{ctx},\mathrm{KVvid}\_n)&\xrightarrow{\mathrm{DiT}}(\mathrm{lat}\_{n+1},\mathrm{KVvid}\_{n+1}),\\\\ \mathrm{lat}\_{n+1}&\xrightarrow{\mathrm{VAE}}\mathrm{frame}\_{n+1}. \end{aligned}$ | $\displaystyle \mathcal Q:(W\_{\rm DiT},W\_{\rm T5},\mathrm{KV})\mapsto(\hat W\_{\rm DiT},\hat W\_{\rm T5},\widehat{\mathrm{KV}})$ | $\displaystyle \begin{aligned}&G=(V,E)\\\\&\mathcal O:\ G\mapsto G'\\\\&\mathcal C\_{\rm compile}:\ G'\xrightarrow{\rm torch.export}G\_{\rm FX}\\\\&G\_{\rm FX}\xrightarrow{\rm AOTInductor}\mathrm{package}\end{aligned}$ | $\displaystyle \Phi\_{\rm pos}:(\mathrm{pos},\theta)\mapsto(\mathrm{pos}^{\prime},\theta^{\prime})$ | $\displaystyle \begin{aligned}&\underbrace{G=(V,E)}\_{\text{computation graph}},\quad\underbrace{\mathcal R\_i\subseteq\mathcal R}\_{\text{resources}}\\\\&\mathrm{plan}^{\star}\in\operatorname\*{arg\\,min}\_{\mathrm{plan}}\widehat t\_{\rm end}(\mathrm{plan})\\\\&M\_{\rm VRAM}\le B\_{\rm VRAM},\quad M\_{\rm RAM}\le B\_{\rm RAM}\end{aligned}$ | $\displaystyle \Phi\_{\rm continue}:(\epsilon\_{n+1},\mathrm{ctx},\mathrm{KVvid}\_n)\mapsto(\mathrm{lat}\_{n+1},\mathrm{KVvid}\_{n+1})$ | $\displaystyle \Phi\_{\rm scene}:(\mathrm{KVvid}\_0,\mathrm{KVtext}\_0,\mathrm{prompt}\_1)\mapsto\begin{cases} (\varnothing,\mathrm{KVtext}\_1),&\text{new scene},\\\\ \Phi(\mathrm{KVvid}\_0,\mathrm{KVtext}\_0),&\text{continuation}. \end{cases}$ |

## MVP1

| Version | Model | Quantization | Compilation | Resolution change | Offloading | Video continuation | Scene change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [MVP1](../../../%D0%98%D1%81%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5.canvas)<br>[Live 1](../../../configs/live.json) | $\displaystyle \begin{aligned}&\mathrm{T5}=\mathrm{UMT5\mbox{-}XXL}\\\\&\mathrm{DiT}:\ \mathrm{Wan}\mapsto\mathrm{LongLive\\,2.0\mbox{-}5B\ S2}\\\\&\mathrm{decoder}:\ \mathrm{Wan\ VAE}\mapsto\mathrm{TAEHV}\end{aligned}$<br>Experiments | $\displaystyle \begin{aligned}&\mathcal Q^{(1)}:\\\\&W\_{\rm DiT}^{\rm BF16}\mapsto\widehat W\_{\rm DiT}^{\rm W4A4}&&\substack{\mathrm{FPS}:6.28\to11.29\\\\\mathrm{FP8}\to\mathrm{W4A4},\ 512\times928}\\\\&W\_{\rm T5}^{\rm BF16}\mapsto\widehat W\_{\rm T5}^{\rm INT8}&&T\_{\rm prompt}:2.86\to1.63\ \mathrm{s}\\\\&\mathrm{KV}^{\rm BF16}\mapsto\widehat{\mathrm{KV}}^{\rm FP8\ E4M3}&&\substack{M\_{\rm KV}:9.819\to4.985\ \mathrm{GiB}\\\\704\times1280}\end{aligned}$<br>FPS and T5 · KV size | $\displaystyle \begin{aligned}&\mathcal O^{(1)}:\\\\&\mathrm{Attn}\_{\rm self}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\\,2}&&\substack{T\_{\rm block}:186.8\to77.3\ \mathrm{ms}\\\\\mathrm{FP8},\ 704\times1280}\\\\&\mathrm{Attn}\_{\rm cross}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\\,2}\\\\&\underbrace{(W\_Q,W\_K,W\_V)\mapsto W\_{\rm QKV}}\_{\text{self-attention}}\\\\&\mathrm{FFN}\mapsto\underbrace{\mathrm{GEMM}\_1\to\mathrm{GELU}\to\mathrm{Quant}\_4}\_{\text{fuse}}\to\mathrm{GEMM}\_2\\\\&\mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}\_{\rm Sage}\end{aligned}$<br>[In detail](../../../%D0%98%D1%81%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5.canvas) · Measurement | $\displaystyle \begin{aligned}&704\times1280\mapsto512\times928\\\\&\mathrm{PI},\quad t=1\\\\&N\_{\rm steps}=2,\quad\mathrm{shift}=5\end{aligned}$<br>[[RoPE-positions]] | $\displaystyle \begin{aligned}&\mathrm{policy}\_1=\mathrm{greedy\\_ram}\\\\&B\_{\rm VRAM}^{\rm margin}=1.5\ \mathrm{GiB}\\\\&B\_{\rm RAM}^{\rm margin}=0\ \mathrm{GiB}\\\\&B\_{\rm T5}^{\rm first}=1.0\ \mathrm{GiB}\end{aligned}$<br>[[Offloading]] | $\displaystyle \Phi\_{\rm continue}^{(1)}:=\left.\Phi\_{\rm continue}\right\rvert\_{\substack{F\_{\rm chunk}=8,\ F\_{\rm anch}=8\\\\F\_{\rm win}=32}}$<br>[[Long-generation]] | $\displaystyle \Phi\_{\rm scene}^{(1)}:=\left.\Phi\_{\rm scene}\right\rvert\_{F\_{\rm scene\ offset}=8}$<br>Mode: not set.<br>[[Scene-change]] |
