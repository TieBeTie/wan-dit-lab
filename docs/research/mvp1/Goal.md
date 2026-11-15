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
| General model | $\displaystyle \begin{aligned} \mathrm{prompt}_j&\xrightarrow{\mathrm{T5}}\mathrm{ctx}_j,\qquad \mathrm{KVtext}_j=\mathrm{KV}^{\rm cross}(\mathrm{ctx}_j),\\ (\epsilon_{n+1},\mathrm{ctx},\mathrm{KVvid}_n)&\xrightarrow{\mathrm{DiT}}(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1}),\\ \mathrm{lat}_{n+1}&\xrightarrow{\mathrm{VAE}}\mathrm{frame}_{n+1}. \end{aligned}$ | $\displaystyle \mathcal Q:(W_{\rm DiT},W_{\rm T5},\mathrm{KV})\mapsto(\hat W_{\rm DiT},\hat W_{\rm T5},\widehat{\mathrm{KV}})$ | $\displaystyle \begin{aligned}&G=(V,E)\\&\mathcal O:\ G\mapsto G'\\&\mathcal C_{\rm compile}:\ G'\xrightarrow{\rm torch.export}G_{\rm FX}\\&G_{\rm FX}\xrightarrow{\rm AOTInductor}\mathrm{package}\end{aligned}$ | $\displaystyle \Phi_{\rm pos}:(\mathrm{pos},\theta)\mapsto(\mathrm{pos}^{\prime},\theta^{\prime})$ | $\displaystyle \begin{aligned}&\underbrace{G=(V,E)}_{\text{computation graph}},\quad\underbrace{\mathcal R_i\subseteq\mathcal R}_{\text{resources}}\\&\mathrm{plan}^{\star}\in\operatorname*{arg\,min}_{\mathrm{plan}}\widehat t_{\rm end}(\mathrm{plan})\\&M_{\rm VRAM}\le B_{\rm VRAM},\quad M_{\rm RAM}\le B_{\rm RAM}\end{aligned}$ | $\displaystyle \Phi_{\rm continue}:(\epsilon_{n+1},\mathrm{ctx},\mathrm{KVvid}_n)\mapsto(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})$ | $\displaystyle \Phi_{\rm scene}:(\mathrm{KVvid}_0,\mathrm{KVtext}_0,\mathrm{prompt}_1)\mapsto\begin{cases} (\varnothing,\mathrm{KVtext}_1),&\text{new scene},\\ \Phi(\mathrm{KVvid}_0,\mathrm{KVtext}_0),&\text{continuation}. \end{cases}$ |

## MVP1

| Version | Model | Quantization | Compilation | Resolution change | Offloading | Video continuation | Scene change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [MVP1](Исследование.canvas)<br>[Live 1](configs/live.json) | $\displaystyle \begin{aligned}&\mathrm{T5}=\mathrm{UMT5\mbox{-}XXL}\\&\mathrm{DiT}:\ \mathrm{Wan}\mapsto\mathrm{LongLive\,2.0\mbox{-}5B\ S2}\\&\mathrm{decoder}:\ \mathrm{Wan\ VAE}\mapsto\mathrm{TAEHV}\end{aligned}$<br>[Experiments](notes/research/Модель.md) | $\displaystyle \begin{aligned}&\mathcal Q^{(1)}:\\&W_{\rm DiT}^{\rm BF16}\mapsto\widehat W_{\rm DiT}^{\rm W4A4}&&\substack{\mathrm{FPS}:6.28\to11.29\\\mathrm{FP8}\to\mathrm{W4A4},\ 512\times928}\\&W_{\rm T5}^{\rm BF16}\mapsto\widehat W_{\rm T5}^{\rm INT8}&&T_{\rm prompt}:2.86\to1.63\ \mathrm{s}\\&\mathrm{KV}^{\rm BF16}\mapsto\widehat{\mathrm{KV}}^{\rm FP8\ E4M3}&&\substack{M_{\rm KV}:9.819\to4.985\ \mathrm{GiB}\\704\times1280}\end{aligned}$<br>[FPS and T5](notes/research/cards/E30.%20%D0%A1%D1%82%D0%B0%D1%80%D1%82%D0%BE%D0%B2%D0%B0%D1%8F%20%D1%82%D0%BE%D1%87%D0%BA%D0%B0%20W4A4%20704%C3%971280%20%D0%B8%20%D0%BE%D1%82%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%88%D0%B0%D0%B3%D0%B8.md) · [KV size](docs/bf16-native.md) | $\displaystyle \begin{aligned}&\mathcal O^{(1)}:\\&\mathrm{Attn}_{\rm self}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\,2}&&\substack{T_{\rm block}:186.8\to77.3\ \mathrm{ms}\\\mathrm{FP8},\ 704\times1280}\\&\mathrm{Attn}_{\rm cross}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\,2}\\&\underbrace{(W_Q,W_K,W_V)\mapsto W_{\rm QKV}}_{\text{self-attention}}\\&\mathrm{FFN}\mapsto\underbrace{\mathrm{GEMM}_1\to\mathrm{GELU}\to\mathrm{Quant}_4}_{\text{fuse}}\to\mathrm{GEMM}_2\\&\mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}_{\rm Sage}\end{aligned}$<br>[In detail](Исследование.canvas) · [Measurement](notes/research/cards/E29.%20%D0%92%D0%B5%D1%80%D1%88%D0%B8%D0%BD%D0%B0%20A%E2%80%B2%20%D0%B8%20%D1%81%D0%BF%D1%83%D1%81%D0%BA%20B,%20FP8,%20704%C3%971280.md) | $\displaystyle \begin{aligned}&704\times1280\mapsto512\times928\\&\mathrm{PI},\quad t=1\\&N_{\rm steps}=2,\quad\mathrm{shift}=5\end{aligned}$<br>[[RoPE-positions]] | $\displaystyle \begin{aligned}&\mathrm{policy}_1=\mathrm{greedy\_ram}\\&B_{\rm VRAM}^{\rm margin}=1.5\ \mathrm{GiB}\\&B_{\rm RAM}^{\rm margin}=0\ \mathrm{GiB}\\&B_{\rm T5}^{\rm first}=1.0\ \mathrm{GiB}\end{aligned}$<br>[[Offloading]] | $\displaystyle \Phi_{\rm continue}^{(1)}:=\left.\Phi_{\rm continue}\right\rvert_{\substack{F_{\rm chunk}=8,\ F_{\rm anch}=8\\F_{\rm win}=32}}$<br>[[Long-generation]] | $\displaystyle \Phi_{\rm scene}^{(1)}:=\left.\Phi_{\rm scene}\right\rvert_{F_{\rm scene\ offset}=8}$<br>Mode: not set.<br>[[Scene-change]] |
