<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# MVP2

[README](../../README.md)

<!-- card mvp2-notation -->
Consider downscaling the frame of the trained model. We compare the clip of a configuration with the clip of the $704\times1280$ reference with the same seed.

$$
\underbrace{\mathrm{cfg}}_{\text{configuration}},\quad
\underbrace{\mathrm{cfg}_0}_{\text{reference }704\times1280},\quad
\underbrace{s}_{\text{seed}},\quad
\underbrace{\mathrm{video}(\mathrm{cfg},s)}_{\text{clip}}.
$$
$$
\underbrace{\mathcal L=\mathrm{LPIPS}_{\rm align}+\bigl(1-\mathrm{DINO}\bigr)}_{\text{loss of the clip to the reference with the same seed}},\quad
\underbrace{\mathcal L_1}_{\text{loss of the first chunk}},\quad
\underbrace{\overline{\mathcal L}}_{\text{mean over 3 seeds and 4 scenes}}.
$$
$$
\underbrace{H\times W}_{\text{frame}},\quad
\underbrace{h\times w=\tfrac{H}{32}\times\tfrac{W}{32}}_{\text{token grid of the frame}},\quad
\underbrace{F_{\rm chunk},F_{\rm anch},F_{\rm win}}_{\text{latent frames: chunk, anchor, window}},\quad
\underbrace{T_{\rm block}}_{\text{time of one DiT block}}.
$$

Now consider what we measure the loss with and which configuration we descend from.

<!-- card mvp2-start -->
# Starting point · MVP2

Which configuration should the descent start from? Consider a configuration as a set of factors; a step changes one factor.

$$
\underbrace{\mathrm{cfg}=(W_{\rm DiT},W_{\rm T5},\mathrm{KV},\mathrm{Attn},\mathrm{RoPE},H\times W)}_{\text{factors}},\qquad
\underbrace{\mathrm{cfg}^{(k)}\mapsto\mathrm{cfg}^{(k+1)}}_{\text{change one factor}},\qquad
\underbrace{704\times1280,\quad\mathrm{KV}=\mathrm{FP8},\quad W_{\rm T5}=\mathrm{BF16}}_{\text{common to all steps}}
$$

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{Attn}\_{\rm self}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\\,2}$ | $\displaystyle \begin{aligned}&\text{Consider vertex A′: FP8, SDPA.}\\\\&\mathrm{Attn}\_{\rm self}=\mathrm{Sage\\,2},\quad\mathrm{RoPE}=\text{separate LongLive kernel}\end{aligned}$ | We change only the self-attention kernel. | $\displaystyle \begin{aligned}&\mathrm{FPS}:1.30\to2.17\\\\&\underbrace{24\text{–}27\ \mathrm{dB}}\_{\text{PSNR of the first chunk to A′}}\end{aligned}$<br>E29 |
| <span class="wan-kept">Applied</span><br>$\displaystyle W\_{\rm DiT}:\ \mathrm{FP8}\mapsto\mathrm{W4A4}$ | $\displaystyle \begin{aligned}&\text{Weights and activations of the linear layers in 4 bits.}\\\\&\underbrace{\mathrm{LongLive\\,2.0\ S2}}\_{\text{trained and distilled in NVFP4}}\end{aligned}$ | For this checkpoint, closer to FP32 does not mean better video. | $\displaystyle \begin{aligned}&\mathrm{FPS}:2.17\to5.41\\\\&\text{By the grid: not worse than FP8.}\end{aligned}$<br>E30 |
| $\displaystyle W\_{\rm T5}:\ \mathrm{BF16}\mapsto\mathrm{INT8}$ | $\displaystyle \begin{aligned}&\text{We quantize the text encoder.}\\\\&\underbrace{T\_{\rm prompt}}\_{\text{prompt change time}}\end{aligned}$ | Faster prompt change. Not applied: postponed to the end. | $\displaystyle \begin{aligned}&\mathrm{FPS}:5.41\to5.58\\\\&T\_{\rm prompt}:2.86\to1.63\ \mathrm{s}\\\\&\text{User: the prompt is matched worse.}\end{aligned}$<br>E30 |
| $\displaystyle \mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}\_{\rm Sage}$ | $\displaystyle \begin{aligned}&\text{Fused kernel with LongLive rounding.}\\\\&\underbrace{0.029\\%}\_{\text{difference from the start per call}}\end{aligned}$ | We remove the separate RoPE pass. Not applied: gives no speed. | $\displaystyle \begin{aligned}&\mathrm{FPS}:5.41\to5.45\\\\&\text{The video diverges like another seed.}\end{aligned}$<br>E30 |

$$
\underbrace{\mathrm{cfg}_{\rm start}=\mathrm{point\mbox{-}b2}}_{\text{W4A4, Sage 2, T5 BF16, KV FP8, }704\times1280},\qquad\mathrm{FPS}=5.41.
$$

Starting point config

<!-- card mvp2-rope -->
# Resolution change and RoPE · MVP2

How to reduce the token grid with the smallest loss? Consider the RoPE frequencies of one spatial axis.

$$
\underbrace{(h,w):(22,40)\mapsto(h',w')}_{\text{token grid of the frame}},\qquad
\underbrace{\theta_j=10000^{-2j/42},\quad j=0,\ldots,20}_{\text{RoPE frequencies of the axis}},\qquad
\underbrace{M_H,M_W}_{\text{multipliers for height and width}}
$$
$$
\theta'_{a,j}=
\begin{cases}
M_a\,\theta_j,&\mathrm{PI},\\
M_a^{\,j/20}\,\theta_j,&\mathrm{NTK},\\
M_a^{\,\mu+(1-\mu)j/20}\,\theta_j,&\mathrm{mix}(\mu),
\end{cases}
\qquad a\in\{H,W\}.
$$

We do not change the time axis. Overview of papers — [[Позиции RoPE|RoPE positions]].

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| $\displaystyle \theta\_j\mapsto M^{\\,j/20}\theta\_j$<br>NTK<br>[YaRN, App. A.2](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\text{We keep the high frequencies and stretch the low ones.}\\\\&\underbrace{M^\star(h')=\arg\min\_M\mathcal L}\_{\text{best }M\text{ at the step}}\end{aligned}$ | We check whether the best $M$ grows as the grid shrinks. Not applied. | $\displaystyle \begin{aligned}&\underbrace{M^\star\approx1.2\text{–}1.3}\_{672,\ 640,\ 608}\\\\&\text{Almost constant, does not grow.}\\\\&\underbrace{\overline{\mathcal L}\_1=0.436}\_{\text{held-out scenes, }M=1.2}\end{aligned}$<br>Descent 09.11 |
| $\displaystyle \mathrm{shift}\mapsto\mathrm{scaleArea}\cdot\mathrm{shift}$<br>[Scaling Rectified Flow, (23)](https://arxiv.org/html/2403.03206#S5.SS3.SSS2) | $\displaystyle \begin{aligned}&\text{We change the noise schedule for a smaller frame.}\\\\&576\times1056:\quad 5\mapsto4.11\mapsto3.63\end{aligned}$ | Fewer tokens, smaller shift. Not applied. | $\displaystyle \begin{aligned}&\text{There is no stable improvement.}\\\\&\mathrm{shift}=5\end{aligned}$ |
| $\displaystyle (M\_H,M\_W,\mu)\mapsto\arg\min\overline{\mathcal L}\_1$<br>CMA-ES | $\displaystyle \begin{aligned}&\text{We search for the profile on the training scenes and check on the held-out ones.}\\\\&\underbrace{\text{cup, fox}}\_{\text{training}},\quad\underbrace{\text{astronaut, two foxes}}\_{\text{held-out}}\\\\&\underbrace{144\ \text{points}}\_{\text{seeds 0 and 1}},\quad\underbrace{\text{seed 2}}\_{\text{held-out}}\end{aligned}$ | Is there a profile better than one number per axis. Not applied: fitting to the training scenes. | $\displaystyle \begin{aligned}&(M\_H,M\_W,\mu)=(1.134,\ 1.104,\ 1)\\\\&\underbrace{0.338}\_{\text{training}},\quad\underbrace{0.382}\_{\text{held-out}}\\\\&\text{Plateau from the third generation.}\end{aligned}$<br>Report, section 14 |
| <span class="wan-kept">Applied</span><br>$\displaystyle M\_H=\frac{22}{h'},\qquad M\_W=\frac{40}{w'}$<br>PI by the grid ratio<br>[Position Interpolation, §2.3](https://arxiv.org/html/2306.15595) | $\displaystyle \begin{aligned}&\text{We stretch the positions of the downscaled frame to the trained grid.}\\\\&\mathrm{pos}\mapsto M\_a\\,\mathrm{pos},\qquad\underbrace{M\_a(h'-1)\approx h-1}\_{\text{the same range of positions}}\\\\&\underbrace{608\times1120:\ M\_H=\tfrac{22}{19},\ M\_W=\tfrac{40}{35}}\_{\text{check on held-out scenes}}\end{aligned}$ | No tuning: the multiplier is set by the grids. The object keeps its position and scale in the frame. | $\displaystyle \begin{aligned}&\underbrace{0.387}\_{\text{held-out, }608}\ \text{versus}\ \underbrace{0.382}\_{\rm CMA}\\\\&\underbrace{0.442}\_{\text{PI}},\ \underbrace{0.619}\_{\text{no interpolation}},\ \underbrace{0.607}\_{\text{uniform }1.1}\quad(512,\ \text{seed }0)\\\\&\text{User: the objects stand where they are in the reference.}\end{aligned}$ |
| $\displaystyle 17\times30\mapsto17\times31$<br>Uniform stretching | $\displaystyle \begin{aligned}&\underbrace{544\times960:\ M\_H=1.294,\ M\_W=1.333}\_{\text{non-uniform}}\\\\&\underbrace{544\times992:\ M\_H=1.294,\ M\_W=1.290}\_{\text{as in the reference}}\end{aligned}$ | We check whether different stretching of the axes spoils the loss. Not applied. | $\displaystyle \begin{aligned}&\overline{\mathcal L}\_1:\quad\underbrace{0.426}\_{544\times960},\quad\underbrace{0.436}\_{544\times992}\\\\&\mathrm{FPS}:-6\\%\end{aligned}$<br>Report, section 30 |

**Resolution ladder**

PI by the grid ratio, $\Sigma_{\rm ordered}$, FPS of the last chunk of the fox with fresh noise, before Sage 2 in cross-attention.

| $H\times W$ | $h\times w$ | FPS, $F\_{\rm win}=32$ | FPS, $F\_{\rm win}=24$ | $\overline{\mathcal L}\_1$ | $\overline{\mathcal L}\_1/\mathcal L\_{\rm ceil}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $704\times1280$ | $22\times40$ | $5.71$ | $6.58$ | $0$ | $0$ |
| $608\times1120$ | $19\times35$ | $8.51$ | $9.61$ | $0.360$ | $0.49$ |
| $576\times1056$ | $18\times33$ | $9.90$ | $11.14$ | $0.393$ | $0.53$ |
| $544\times992$ | $17\times31$ | $11.83$ | $13.20$ | $0.414$ | $0.56$ |
| $544\times960$ | $17\times30$ | $12.53$ | $13.94$ | $0.426$ | $0.58$ |
| $512\times928$ | $16\times29$ | $14.18$ | $15.73$ | $0.483$ | $0.66$ |

The step $544\mapsto512$ costs $0.069$ of loss; the previous steps cost $0.02$–$0.03$.

Grids · NTK hypothesis · Frequency-aware

<!-- card mvp2-memory -->
# Video memory · MVP2

Can the video memory be made cheaper without losing the scene? Consider the generation of one chunk: two denoising steps and a write pass.

$$
(\epsilon_{n+1},\mathrm{ctx},\mathrm{KVvid}_n)
\xrightarrow{\ t_1\ }\widehat{\mathrm{lat}}_{n+1}
\xrightarrow{\ t_2=833\ }\mathrm{lat}_{n+1}
\xrightarrow{\ t=0,\ \text{write}\ }\mathrm{KVvid}_{n+1}
$$
$$
\underbrace{T_{\rm chunk}\approx3\cdot30\cdot T_{\rm block}}_{\text{three passes over 30 blocks}},\qquad
\underbrace{\mathrm{KVvid}_n=\mathrm{KVanch}\oplus\mathrm{KVupd}_n}_{\text{anchor and ring}},\qquad
\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)}_{\text{MVP1}}
$$

MVP1 memory rules — [[Длительная генерация|Long generation]].

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| $\displaystyle F\_{\rm win}:32\mapsto24$ | $\displaystyle \begin{aligned}&\text{The ring is shorter by one chunk.}\\\\&\underbrace{F\_{\rm win}-F\_{\rm anch}-F\_{\rm chunk}:\ 16\mapsto8}\_{\text{frames in the ring}}\\\\&\text{The window takes effect when the ring is full.}\end{aligned}$ | Fewer keys in self-attention. Not applied: risk of drift and a failure at a scene change. | $\displaystyle \begin{aligned}&\mathrm{FPS}:+12\text{–}15\\%\\\\&\text{The first three chunks are bit-for-bit equal.}\\\\&\underbrace{\lvert\Delta B\rvert>15}\_{\text{blue shift of the fox, 12 chunks}}:\ \underbrace{3\ \text{of}\ 12}\_{F\_{\rm win}=24},\quad\underbrace{0\ \text{of}\ 12}\_{F\_{\rm win}=32}\\\\&\text{Fails at a scene change: the ring equals one chunk.}\end{aligned}$<br>Report, section 27 |
| $\displaystyle F\_{\rm win}:32\mapsto16$ | $\displaystyle \begin{aligned}&\text{No ring: anchor and new chunk.}\\\\&16=8+0+8\end{aligned}$ | The limit of shrinking the window. Not applied. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:+27\\%}\_{608\times1120}\\\\&\underbrace{3.86\ /\ 4.68}\_{\text{seam: cup / fox}}\\\\&\text{The cup empties, the frame resets.}\end{aligned}$ |
| $\displaystyle \mathrm{KVnew}\_{n+1}:\ t=0\mapsto t\_2$<br>`cache_write: last_step` | $\displaystyle \begin{aligned}&\text{We take K/V from the last denoising step.}\\\\&T\_{\rm chunk}:\ 3\cdot30\\,T\_{\rm block}\mapsto2\cdot30\\,T\_{\rm block}\\\\&\underbrace{\cos\bigl(V^{t\_2},V^{t=0}\bigr)}\_{\text{similarity to the clean write, per block}}\end{aligned}$ | We remove a third of the chunk. Not applied: the video memory breaks. | $\displaystyle \begin{aligned}&\underbrace{\cos V=0.43,\quad\cos K=0.76}\_{t\_2=833}\\\\&\underbrace{\mathrm{FPS}:8.60\to11.25}\_{608\times1120}\\\\&\text{Water in the cup does not fill;}\\\\&\text{the fox's camera drifts away.}\end{aligned}$<br>Report, part 3 |
| $\displaystyle t\_2:833\mapsto t$<br>Write at low noise | $\displaystyle \begin{aligned}&\text{We add noise to the clean chunk latent up to }t	ext{ and compare K/V.}\\\\&t\in\\{600,400,250,100\\}\end{aligned}$ | Is there a step time at which K/V are close to the clean ones. Not applied. | $\displaystyle \begin{aligned}&\cos V:\quad0.48,\ 0.58,\ 0.66,\ 0.70\\\\&\text{K/V depend on the modulation by }t	ext{.}\end{aligned}$ |
| $\displaystyle \mathrm{KVvid}\_n\mapsto\mathrm{KVanch}$<br>in the write pass<br>`clean_pass_sink` | $\displaystyle \begin{aligned}&\text{In the write pass, attention only on the anchor and the chunk.}\\\\&\underbrace{T\_{\rm Sage}:12.0\to4.1\ \mathrm{ms}}\_{\text{estimate from the first chunk}}\end{aligned}$ | We make the third pass cheaper. Not applied. | $\displaystyle \begin{aligned}&\mathrm{FPS}:-1.4\\%\\\\&\text{The mask adds its own overhead.}\end{aligned}$ |
| <span class="wan-kept">Applied</span><br>$\displaystyle \Phi\_{\rm continue}^{(2)}=\Phi\_{\rm continue}^{(1)}$ | $\displaystyle \begin{aligned}&(F\_{\rm chunk},F\_{\rm anch},F\_{\rm win})=(8,8,32)\\\\&\text{A separate write pass at }t=0	ext{.}\end{aligned}$ | We keep the MVP1 video memory unchanged. | $\displaystyle \begin{aligned}&\underbrace{0\ \text{of}\ 12}\_{\text{strong color drift}}\\\\&\underbrace{1.71\ /\ 1.13}\_{\text{seam: fox / cup, }512}\end{aligned}$ |

Window grids · Report 11–12.11

<!-- card mvp2-speed -->
# Compilation and placement · MVP2

What limits the chunk speed and what speeds up without changing the result? Consider the chunk time and the composition of one block.

$$
\underbrace{T_{\rm chunk}=3\cdot30\cdot T_{\rm block}+T_{\rm dec}}_{\text{three passes and the decoder}},\qquad
\underbrace{T_{\rm block}=T_{\rm GEMM}+T_{\rm Sage}+T_{\rm prep}+T_{\rm norm}+\cdots}_{\text{kernels of one block}},\qquad
\underbrace{\mathrm{FPS}=\frac{4\,F_{\rm chunk}}{T_{\rm chunk}}}_{\text{4 frames per latent frame}}
$$

$$
\underbrace{T_{\rm block}=24.3\ \mathrm{ms}}_{512\times928,\ F_{\rm win}=32}:\qquad
\underbrace{38.6\%}_{\text{linear W4A4}},\quad
\underbrace{37.9\%}_{\text{Sage self and cross}},\quad
\underbrace{7.5\%}_{\text{Sage preparation}},\quad
\underbrace{6.8\%}_{\text{norms, modulation}},\quad
\underbrace{3.6\%}_{\text{activation quantization}}
$$

Graph and fusions of MVP1 — [Compilation · MVP1](../../MVP1%20EN.canvas). Placement — [[Offloading]].

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{Attn}\_{\rm cross}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\\,2}$ | $\displaystyle \begin{aligned}&\text{Consider the cross-attention of the block.}\\\\&\underbrace{\mathrm{KVtext}}\_{\text{computed once per prompt}}\end{aligned}$ | The same kernel as in self-attention. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:14.18\to14.71}\_{512\times928,\ +3.7\\%}\\\\&\Delta\overline{\mathcal L}\_1=+0.001\end{aligned}$<br>Report, part 3 |
| <span class="wan-kept">Applied</span><br>$\displaystyle \Sigma\_{\rm ordered}\mapsto\Sigma\_{\rm atomic}$<br>Live only | $\displaystyle \begin{aligned}&\text{In Live a bit-exact repeat is not needed.}\\\\&B\_{\rm VRAM}^{\rm margin}:0.0625\mapsto0\end{aligned}$ | We return the fast summation order and give back the VRAM margin. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:14.71\to14.82\to15.05}\_{512\times928}\\\\&\underbrace{13.08\to13.44}\_{544\times960}\end{aligned}$ |
| $\displaystyle \mathrm{Attn}\_{\rm self}\mapsto\mathrm{Attn}\_{\rm sparse}$<br>`kernels.sparse` | $\displaystyle \begin{aligned}&\text{We drop the keys with a small contribution.}\\\\&\underbrace{\varepsilon=0.02,\ \text{anchor}+8}\_{\text{variant 1}},\qquad\underbrace{\text{anchor}+\text{last 4}}\_{\text{variant 2}}\end{aligned}$ | Fewer keys in the Sage kernel. Not applied: with window 24 there are only three chunks of keys. | $\displaystyle \begin{aligned}&\underbrace{+2.8\\%,\quad\Delta\overline{\mathcal L}=+0.023}\_{\text{variant 1}}\\\\&\underbrace{+1.0\\%}\_{\text{variant 2}}\end{aligned}$ |
| $\displaystyle T\_{\rm dec}\parallel T\_{\rm block}$ | $\displaystyle \begin{aligned}&\text{The decoder on its own stream on top of the next chunk.}\\\\&T\_{\rm dec}=8\cdot12.7\ \mathrm{ms}\end{aligned}$ | We overlap the decoder with the blocks. Not applied: the GPU is already busy computing. | $\displaystyle \begin{aligned}&\text{The latent is bit-for-bit equal.}\\\\&\mathrm{FPS}:13.44\to13.37\\\\&M\_{\rm VRAM}=8.07\ \text{GB}\end{aligned}$<br>Report, section 30 |
| $\displaystyle \mathrm{prep}(K\_{\rm win}):\ 3\mapsto1$<br>per chunk | $\displaystyle \begin{aligned}&\text{The window of past chunks does not change between passes.}\\\\&\underbrace{\overline K,\ \operatorname{scale}(V)}\_{\text{over all keys, including the new chunk}}\end{aligned}$ | Do not recompute the preparation of the window keys. Not applied: impossible bit-for-bit. | $\displaystyle \begin{aligned}&\underbrace{2.2\ \text{GB}}\_{\text{cache of the prepared window in VRAM}}\\\\&\underbrace{\le3.6\\%}\_{\text{estimate}}\end{aligned}$ |
| $\displaystyle \mathrm{Norm}\to\mathrm{Quant}\_4\mapsto\mathrm{fused}$ | $\displaystyle \begin{aligned}&\text{Norm, modulation and quantization in one kernel.}\\\\&0.86+0.79\ \mathrm{ms}\ \text{per block}\end{aligned}$ | We remove the write and the repeated read of BF16. Not done. | $\displaystyle \begin{aligned}&\underbrace{\approx2\\%}\_{\text{estimate}}\\\\&\text{A new kernel in the Nunchaku code.}\end{aligned}$ |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{policy}\_2=\mathrm{greedy\\_vram}$ | $\displaystyle \begin{aligned}&\text{Consider the delivery of the weight rows.}\\\\&\underbrace{7.1\ \mathrm{ms}}\_{\text{row copy RAM → VRAM}}<\underbrace{T\_{\rm block}}\_{\text{hidden behind the computation}}\end{aligned}$ | We check whether the weight transfer is the limit. | $\displaystyle \begin{aligned}&\underbrace{1\ \mathrm{ms}}\_{\text{waiting for copies per run}},\quad\underbrace{0}\_{\text{late arrivals}}\\\\&\text{The speed is limited by the block computation.}\end{aligned}$<br>Report, section 12 |

$$
\underbrace{T_{\rm chunk}=2.39\ \mathrm{s}}_{544\times960,\ F_{\rm win}=32}
\approx\underbrace{90\cdot25.2\ \mathrm{ms}}_{\text{blocks}}+\underbrace{0.10\ \mathrm{s}}_{\text{decoder}},\qquad
\mathrm{FPS}=\frac{32}{2.39}\approx13.4.
$$

Transformations with a bit-for-bit equal latent are exhausted. Further speed comes only from the linear W4A4 and Sage kernels.

<!-- card mvp2-loss -->
# Loss measurement · MVP2

How well does the loss to the reference distinguish configurations? Consider two clips and the distance between them.

$$
\underbrace{\mathcal L(a,b)}_{\text{loss of clip }a\text{ to clip }b},\qquad
\underbrace{\mathcal L_{\rm floor}=\mathcal L\bigl(\mathrm{video}'(\mathrm{cfg}_0,s),\mathrm{video}(\mathrm{cfg}_0,s)\bigr)}_{\text{repeat of the same configuration}},\qquad
\underbrace{\mathcal L_{\rm ceil}=\mathcal L\bigl(\mathrm{video}(\mathrm{cfg}_0,s'),\mathrm{video}(\mathrm{cfg}_0,s)\bigr)}_{\text{another seed}}
$$
$$
\underbrace{\frac{\mathcal L-\mathcal L_{\rm floor}}{\mathcal L_{\rm ceil}-\mathcal L_{\rm floor}}}_{\text{share of the gap between the repeat and another clip}}
$$

| Transformation                                                                                                                                          | Context → computation                                                                                                                                                                                                                                                                                                          | Idea                                                                                                                  | Observations                                                                                                                                                                                                                                                                                                                                          |
| :------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\displaystyle \mathrm{video}\_0\mapsto\operatorname{resize}\_{608}(\mathrm{video}\_0)$                                                                    | $\displaystyle \begin{aligned}&\text{Downscale the finished reference by area, without generation.}\\\\&\mathcal L\bigl(\operatorname{resize}\_{608}(\mathrm{video}\_0),\mathrm{video}\_0\bigr)\end{aligned}$                                                                                                                            | The price of the pixel loss itself.                                                                                           | $\displaystyle \begin{aligned}&\underbrace{0.013}\_{\text{fox}},\quad\underbrace{0.008}\_{\text{cup}}\\\\&\text{3 chunks, seed 0}\end{aligned}$                                                                                                                                                                                                  |
| $\displaystyle \mathrm{cfg}\_0\mapsto\mathrm{cfg}\_0$<br>Repeat without changes.                                                                            | $\displaystyle \begin{aligned}&\text{Consider float summation from parallel kernel tiles.}\\\\&\underbrace{\Sigma\_{\rm atomic}}\_{\text{order of summands is not fixed}}\\\\&\underbrace{\overline K}\_{\text{mean of the keys before Sage}},\qquad\underbrace{x\\,W\_{\rm down}}\_{\text{low-rank branch of W4A4}}\end{aligned}$ | We check whether one config gives one clip. Two sources: our key preparation kernel and the Nunchaku kernel.                | $\displaystyle \begin{aligned}&\underbrace{36\text{–}38\\%}\_{\text{divergence of the first chunk latent}}\\\\&\mathcal L\_{\rm floor}:\quad\underbrace{0.26}\_{\text{fox}},\quad\underbrace{0.18}\_{\text{cup}}\\\\&\text{The divergence starts at block 0.}\end{aligned}$<br>Report, section 3 |
| <span class="wan-kept">Applied</span><br>$\displaystyle \Sigma\_{\rm atomic}\mapsto\Sigma\_{\rm ordered}$<br>`kernels.lora_sum`                         | $\displaystyle \begin{aligned}&\text{Each tile writes its own partial sum.}\\\\&\underbrace{p\_1,\ldots,p\_m}\_{\text{partial sums of the tiles}}\\\\&\Sigma\_{\rm ordered}=\bigl((p\_1+p\_2)+\cdots\bigr)+p\_m\\\\&\text{We sum in a fixed order.}\end{aligned}$                                                | One config gives one clip: the repeat is comparable bit by bit. In experiments ordered, in Live atomic.                          | $\displaystyle \begin{aligned}&\mathcal L\_{\rm floor}=0\\\\&\text{The latent and MP4 of two repeats are bit-for-bit equal.}\\\\&\underbrace{T\_{\rm block}:38.0\to38.8\ \mathrm{ms}}\_{608\times1120},\quad\mathrm{FPS}:-2\\%\end{aligned}$<br>Report, section 11                                             |
| $\displaystyle M:1\mapsto1+10^{-5}$<br>`rope.m`                                                                                                         | $\displaystyle \begin{aligned}&\text{One small perturbation, three variants of weights and attention.}\\\\&\underbrace{\delta=\frac{\lVert\mathrm{lat}'-\mathrm{lat}\rVert}{\lVert\mathrm{lat}\rVert}}\_{\text{divergence of the first chunk latent}}\\\\&704\times1280,\quad\text{fox},\quad\Sigma\_{\rm ordered}\end{aligned}$       | Where the amplification of a small error comes from: the two-step model or the 4 bits.                                                     | $\displaystyle \begin{aligned}&\underbrace{\delta=0.35}\_{\mathrm{W4A4}+\mathrm{Sage\\,2}}\\\\&\underbrace{\delta=0.18}\_{\mathrm{FP8}+\mathrm{Sage\\,2}}\\\\&\underbrace{\delta=0.16}\_{\mathrm{FP8}+\mathrm{SDPA}}\\\\&\text{The model itself amplifies; W4A4 doubles it.}\end{aligned}$                                                            |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathcal L\mapsto\overline{\mathcal L}\_1$                                                     | $\displaystyle \begin{aligned}&\text{We average the loss of the first chunk.}\\\\&\underbrace{s\in\\{0,1,2\\}}\_{\text{seed}},\quad\underbrace{4\ \text{scenes}}\_{\text{cup, fox, astronaut, two foxes}}\\\\&\text{Order of the chains as in the reference:}\\\\&\text{the seed noise runs consecutively through the chains.}\end{aligned}$                 | The loss accumulates already in the first chunk; one point costs about 5 s. Averaging removes the dependence on a single trajectory. | $\displaystyle \begin{aligned}&\underbrace{\overline{\mathcal L}\_1=0.360,\quad\overline{\mathcal L}=0.398}\_{608\times1120}\\\\&\underbrace{\mathcal L\_{\rm ceil}=0.736}\_{\text{first chunk}},\quad\underbrace{0.753}\_{\text{clip}}\end{aligned}$<br>Report, section 27                        |
| <span class="wan-kept">Applied</span><br>$\displaystyle \epsilon\mapsto\operatorname{spectral}\_{h\times w}(\epsilon\_{22\times40})$<br>`descent.noise` | $\displaystyle \begin{aligned}&\text{We get the noise of the downscaled frame from the noise of the reference.}\\\\&\underbrace{\epsilon\_{22\times40}}\_{\text{reference noise}}\xrightarrow{\text{truncate the spectrum}}\underbrace{\epsilon\_{h\times w}}\_{\text{noise of the downscaled frame}}\\\\&\text{Computed on the CPU at every chunk.}\end{aligned}$                 | The reference and the downscaled frame share the low frequencies of the noise. For comparison only; in Live the noise is fresh.                     | $\displaystyle \begin{aligned}&\underbrace{0.43\ \mathrm{s}}\_{\text{per chunk}}\\\\&\underbrace{\mathrm{FPS}:8.60\to9.73}\_{608\times1120,\ \text{fresh noise}}\\\\&\text{We measure FPS with fresh noise,}\\\\&\text{the loss with the spectral one.}\end{aligned}$                                                                                        |

Report 11–12.11 · Measurement plan

<!-- card mvp2-boundary -->
---

**Choice of the working point 544×960, window 32 · next stage**

<!-- card mvp2-result -->
# Working point · MVP2

Which configuration do we fix? Consider the candidates near the constraint $\mathrm{FPS}\ge15$.

$$
\min_{\mathrm{cfg}}\ \overline{\mathcal L}_1(\mathrm{cfg})\quad\text{subject to}\quad\mathrm{FPS}\ge15,\quad M_{\rm VRAM}\le8\ \text{GB},\quad M_{\rm RAM}\le16\ \text{GB}.
$$

| Candidate | FPS, Live | $M\_{\rm VRAM}$ | $\overline{\mathcal L}\_1$ | $\overline{\mathcal L}\_1/\mathcal L\_{\rm ceil}$ | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <span class="wan-kept">MVP2 result</span><br>$544\times960,\ F\_{\rm win}=32$ | $13.44$ | $7.1$ GB | $0.426$ | $0.58$ | User's choice by the grids: quality matters more than $1.5$ FPS |
| $544\times960,\ F\_{\rm win}=24$ | $15.03$ | $6.8$ GB | $0.426$ | $0.58$ | Fails at a scene change |
| $512\times928,\ F\_{\rm win}=32$ | $15.05$ | $7.1$ GB | $0.484$ | $0.66$ | Fallback point at 15 FPS |

$$
\begin{aligned}
\mathrm{cfg}_{\rm MVP2}&:\quad H\times W=544\times960,\quad h\times w=17\times30,\\
&\underbrace{M_H=\tfrac{22}{17},\quad M_W=\tfrac{40}{30}}_{\text{PI by the grid ratio}},\quad
\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)}_{\text{MVP1 video memory}},\\
&\underbrace{W_{\rm DiT}=\mathrm{W4A4},\quad W_{\rm T5}=\mathrm{BF16},\quad\mathrm{KV}=\mathrm{FP8}}_{\text{precision}},\quad
\underbrace{\mathrm{Attn}_{\rm self}=\mathrm{Attn}_{\rm cross}=\mathrm{Sage\,2}}_{\text{attention kernels}},\\
&\underbrace{\mathrm{FPS}=13.44<15}_{\text{constraint not met}},\quad\overline{\mathcal L}_1=0.426.
\end{aligned}
$$

The constraint $\mathrm{FPS}\ge15$ is not met at the chosen point. The missing $11.6\%$ we look for in the kernels; we do not change the resolution and the window.

Live config · Experiment config · 512×928

Final grids · Report 11–12.11 · [Version summary](../../Версии%20исследования%20EN.canvas)
