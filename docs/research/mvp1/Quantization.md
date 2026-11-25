<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Quantization · MVP1

How do we fit a 5B model and its video memory into 8 GB? Consider one tensor $X$ and its $b$-bit code.

$$
\underbrace{X}_{\text{weights, activations or K/V}},\qquad
\underbrace{\widehat X=s\cdot\operatorname{round}_{\mathcal G}(X/s)}_{\text{scale }s\text{, rounding to the grid }\mathcal G},\qquad
\underbrace{\mathcal G\in\{\mathrm{FP8\ E4M3},\ \mathrm{INT8},\ \mathrm{INT4}\}}_{\text{grid}},\qquad
\underbrace{M_X\approx\lvert X\rvert\,b/8}_{\text{bytes at }b\text{ bits per element}}
$$

A quantizer is three choices: the grid, the part of the tensor that one scale covers, and what compensates the rounding error.

Now consider where the 16-bit tensors live in the generation of one chunk.

$$
\underbrace{\mathrm{ctx}=\mathrm{T5}(\mathrm{prompt};W_{\rm T5})}_{\text{text encoder, once per prompt}},\qquad
\underbrace{y=xW^{\top}+b,\quad W\in W_{\rm DiT}}_{\text{linear layer of a DiT block}},\qquad
\underbrace{\mathrm{Attn}(Q,\ \mathrm{KVvid}_n\oplus\mathrm{KVnew}_{n+1})}_{\text{self-attention reads the video memory}}
$$
$$
\mathcal Q^{(1)}:\ (W_{\rm DiT},\,W_{\rm T5},\,\mathrm{KVvid})\mapsto(\widehat W_{\rm DiT},\,\widehat W_{\rm T5},\,\widehat{\mathrm{KVvid}})
\qquad
\begin{array}{l|c|c|c}
&\text{was}&\text{MVP1}&\text{one scale covers}\\\hline
W_{\rm DiT}&\mathrm{BF16}&\mathrm{INT4}+\text{rank-32 branch in 16 bits}&\text{group of 64}\\
x&\mathrm{BF16}&\mathrm{INT4}&\text{smoothing }s\text{ per channel}\\
\mathrm{KVvid}&\mathrm{BF16}&\mathrm{FP8\ E4M3}&\text{one token}\\
W_{\rm T5}&\mathrm{BF16}&\mathrm{INT8}&\text{not recorded}
\end{array}
$$

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| Not applied: no speed-up, replaced by W4A4<br>$\displaystyle W_{\rm DiT}:\ \mathrm{BF16}\mapsto\mathrm{FP8}$<br>[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | $\displaystyle \begin{aligned}&\text{Consider the layer }y=xW^{\top}+b.\\[5pt]&\underbrace{\widehat W=s\cdot\mathrm{fp8}(W/s),\quad\widehat x=s_x\cdot\mathrm{fp8}(x/s_x)}_{\text{weights and activations in 8 bits, one scale per tensor}}\\[5pt]&\underbrace{s\mapsto s_i}_{\text{scale per row}}:\ \text{on sm\_89 the kernel applies it to the output only.}\end{aligned}$ | Half the bytes of the weights. The FP8 error is set by the mantissa, not by the scale, so one scale per tensor is enough. | $\displaystyle \begin{aligned}&\underbrace{3.7\%\ \text{vs}\ 3.5\%}_{\text{error: per tensor vs per row, synthetic}}\\[5pt]&\underbrace{23.6\,/\,21.4\,/\,20.9\ \mathrm{dB}}_{\text{PSNR to BF16: first s / middle / last s}}\\[5pt]&\underbrace{\mathrm{FPS}:1.39\to1.37}_{704\times1280,\ \text{2 steps}}\\[5pt]&\text{Per row costs 15–55\% more.}\\[5pt]&\underbrace{M_{\rm peak}:4.77\to2.73\ \mathrm{GiB}}_{256\times448,\ \text{FP8 weights and KV, Self-Forcing}}\end{aligned}$<br>[E19](../../../notes/research/cards/E19.%20FP8%20%D0%BD%D0%B0%20%D1%82%D0%B5%D0%BD%D0%B7%D0%BE%D1%80%20%D0%B8%20%D0%BF%D0%BE%20%D1%81%D1%82%D1%80%D0%BE%D0%BA%D0%B0%D0%BC%2C%20704%C3%971280.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle W_{\rm DiT}:\ \mathrm{FP8}\mapsto\mathrm{W4A4}$<br>[SVDQuant, §3](https://arxiv.org/abs/2411.05007)<br>Kernels: Nunchaku. | $\displaystyle \begin{aligned}&\text{Was: one 8-bit grid carries all of }W.\\[5pt]&\text{Now consider the smoothed weight split into two parts.}\\[5pt]&W\operatorname{diag}(s)=\underbrace{L_1L_2}_{\text{rank 32, 16 bits}}+\underbrace{R}_{\text{residual}}\\[5pt]&y=\underbrace{x\,(L_2/s)^{\top}L_1^{\top}}_{\text{low-rank branch reads the original }x}+\underbrace{Q_4(x/s)\,Q_4(R)^{\top}}_{\text{INT4 GEMM}}+b\\[5pt]&\underbrace{s}_{\text{SmoothQuant, }\alpha=0.5,\ \text{calibrated on 300 layers}}\end{aligned}$ | The largest directions of the weight stay in 16 bits; the 4-bit grid carries only the residual. From the authors: the method and the kernels. Ours: the package, the calibration, the separate extension with the INT4 GEMM. | $\displaystyle \begin{aligned}&\underbrace{T_{\rm GEMM}:4.61\to3.58\to1.30\ \mathrm{ms}}_{\mathrm{BF16}\to\mathrm{FP8}\to\mathrm{W4A4},\ 7040\times3072\times3072}\\[5pt]&\underbrace{\mathrm{FPS}:2.98\to3.64}_{704\times1280,\ \text{2 steps, Sage 2}}\\[5pt]&\underbrace{\mathrm{FPS}:6.28\to11.29}_{512\times928}\\[5pt]&\underbrace{16.1\,/\,16.6\,/\,16.6\ \mathrm{dB}}_{\text{PSNR to BF16; FP8: 19–22}}\\[5pt]&\underbrace{2.78\ \mathrm{GB}}_{\text{30 blocks, 92 MB each}}\\[5pt]&\text{In viewing: sharp and coherent, the scene departs more.}\end{aligned}$<br>[E22](../../../notes/research/cards/E22.%20W4A4%20SVDQuant%20%D0%BD%D0%B0%20%D1%8F%D0%B4%D1%80%D0%B0%D1%85%20Nunchaku%2C%20704%C3%971280.md) · [E30](../mvp2/E30.%20Starting%20point%20W4A4%20704%C3%971280%20and%20postponed%20steps.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{KVvid}:\ \mathrm{BF16}\mapsto\mathrm{FP8\ E4M3}$ | $\displaystyle \begin{aligned}&\text{Consider the stored key of one token, }k_i.\\[5pt]&\underbrace{\widehat k_i=s_i\cdot\mathrm{fp8}(k_i/s_i)}_{\text{one scale per token}},\qquad\widehat v_i\ \text{the same way}\\[5pt]&\underbrace{M_{\rm KV}=30\cdot2\cdot F_{\rm win}N_{\rm frame}\,d\cdot b/8}_{\text{blocks}\times(K,V)\times\text{tokens}\times\text{channels}}+\text{scales}\end{aligned}$ | The video memory is the largest buffer that changes at run time, so it cannot go to disk. One byte per element halves it. | $\displaystyle \begin{aligned}&\underbrace{M_{\rm KV}:9.819\to4.985\ \mathrm{GiB}}_{704\times1280}\\[5pt]&\underbrace{M_{\rm KV}=2.628\ \mathrm{GiB}}_{512\times928,\ \mathrm{FP8}}\\[5pt]&\text{Not measured: FP8 KV alone with BF16 weights.}\end{aligned}$<br>[KV size](../../bf16-native.md) · [H11](../../../notes/research/cards/H11.%20%D0%9A%D0%B2%D0%B0%D0%BD%D1%82%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%82%D0%BE%D0%BB%D1%8C%D0%BA%D0%BE%20KV%20%D1%81%D0%BE%D1%85%D1%80%D0%B0%D0%BD%D0%B8%D1%82%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%BE%20%D0%B8%20%D0%BE%D1%81%D0%B2%D0%BE%D0%B1%D0%BE%D0%B4%D0%B8%D1%82%20%D0%BF%D0%B0%D0%BC%D1%8F%D1%82%D1%8C.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \widehat{\mathrm{KVvid}}\to\mathrm{BF16}\to\mathrm{Sage}\ \mapsto\ \widehat{\mathrm{KVvid}}\to\mathrm{Sage}$ | $\displaystyle \begin{aligned}&\text{Was: the cache is unpacked into a BF16 copy and Sage quantizes it again.}\\[5pt]&\text{Now the kernel reads }\widehat k_i,\widehat v_i\text{ and }s_i\text{ and writes the Sage format directly:}\\[5pt]&\underbrace{K-\operatorname{mean}(K)\to\mathrm{INT8}}_{\text{keys}},\qquad\underbrace{V\to\mathrm{FP8}}_{\text{scale per channel}}\end{aligned}$ | We remove the 16-bit copy of the whole window and one pass over it. | $\displaystyle \begin{aligned}&\underbrace{T_{\rm attn}:42.8\to32.0\ \mathrm{ms}}_{\text{full cache, 28160 keys}}\\[5pt]&\underbrace{T_{\rm block}:73.8\to63.0\ \mathrm{ms}}_{\text{median in a run}}\\[5pt]&M_{\rm VRAM}:6.65\to6.15\ \mathrm{GiB}\\[5pt]&\underbrace{16.1\to16.5\ \mathrm{dB}}_{\text{PSNR to BF16, first third}}\end{aligned}$<br>[E25](../../../notes/research/cards/E25.%20Sage%202%20%D0%BF%D1%80%D1%8F%D0%BC%D0%BE%20%D0%BF%D0%BE%20FP8-%D0%BA%D1%8D%D1%88%D1%83%2C%20704%C3%971280.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle W_{\rm T5}:\ \mathrm{BF16}\mapsto\mathrm{INT8}$ | $\displaystyle \begin{aligned}&\text{Consider the text encoder: it runs once per prompt, not per chunk.}\\[5pt]&\mathrm{ctx}=\mathrm{T5}(\mathrm{prompt};\widehat W_{\rm T5}),\qquad\underbrace{T_{\rm prompt}}_{\text{prompt change time}}\end{aligned}$ | The encoder precision changes the prompt delay and the memory, not the FPS. | $\displaystyle \begin{aligned}&T_{\rm prompt}:2.86\to1.63\ \mathrm{s}\\[5pt]&\underbrace{M_{\rm RAM}:5.96\to5.67\ \mathrm{GiB}}_{\text{pinned}}\\[5pt]&\underbrace{\mathrm{FPS}:5.41\to5.58}_{704\times1280,\ \mathrm{W4A4},\ \text{8 chunks, measured 03.11}}\end{aligned}$<br>[E30](../mvp2/E30.%20Starting%20point%20W4A4%20704%C3%971280%20and%20postponed%20steps.md) |

$$
\mathcal Q^{(1)}=\bigl(\underbrace{\mathrm{W4A4}}_{W_{\rm DiT}},\ \underbrace{\mathrm{INT8}}_{W_{\rm T5}},\ \underbrace{\mathrm{FP8\ E4M3}}_{\mathrm{KVvid}}\bigr)
$$

[MVP1 config](../../../configs/mvp1.json) · [256p FP8 videos](file:///C:/Inner%20Firmament/Growth%20Space/Projects/wan-dit-lab/outputs/experiments/2026-10-17/02_%D0%A0%D0%B0%D1%81%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5_256p_FP8_60%D1%81%D0%B5%D0%BA) · Experiments: [E19](../../../notes/research/cards/E19.%20FP8%20%D0%BD%D0%B0%20%D1%82%D0%B5%D0%BD%D0%B7%D0%BE%D1%80%20%D0%B8%20%D0%BF%D0%BE%20%D1%81%D1%82%D1%80%D0%BE%D0%BA%D0%B0%D0%BC%2C%20704%C3%971280.md) · [E22](../../../notes/research/cards/E22.%20W4A4%20SVDQuant%20%D0%BD%D0%B0%20%D1%8F%D0%B4%D1%80%D0%B0%D1%85%20Nunchaku%2C%20704%C3%971280.md) · [E25](../../../notes/research/cards/E25.%20Sage%202%20%D0%BF%D1%80%D1%8F%D0%BC%D0%BE%20%D0%BF%D0%BE%20FP8-%D0%BA%D1%8D%D1%88%D1%83%2C%20704%C3%971280.md) · [E26](../../../notes/research/cards/E26.%20Live%20W4A4%20%D0%B8%20Sage%202%2C%20480%C3%97832.md) · What changed next — [[docs/research/mvp2/Quantization|Quantization · MVP2]].
