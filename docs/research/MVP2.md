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

Now consider the path from the quality vertex to the starting point; under each point is its FPS at $704\times1280$.

$$
\underbrace{A'}_{\substack{\mathrm{FP8},\ \mathrm{SDPA}\\1.30}}
\xrightarrow{\ \mathrm{Attn}_{\rm self}\ }
\underbrace{B}_{\substack{\mathrm{FP8},\ \mathrm{Sage\,2}\\2.17}}
\xrightarrow{\ W_{\rm DiT}\ }
\underbrace{B_2=\mathrm{cfg}_{\rm start}}_{\substack{\mathrm{W4A4},\ \mathrm{Sage\,2}\\5.41}}
\ \begin{cases}
\xrightarrow{\ W_{\rm T5}\ }B_{2t}:\ 5.58,&\text{rolled back}\\[4pt]
\xrightarrow{\ \mathrm{RoPE\ fuse}\ }B_{2r}:\ 5.45,&\text{closed}
\end{cases}
$$

The precision steps $W_{\rm DiT}$ and $W_{\rm T5}$ — [Quantization · MVP2](../../docs/research/mvp2/Quantization.md).

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{Attn}_{\rm self}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\,2}$ | $\displaystyle \begin{aligned}&\text{Consider vertex A′: FP8, SDPA.}\\[6pt]&\mathrm{Attn}_{\rm self}=\mathrm{Sage\,2},\quad\mathrm{RoPE}=\text{separate LongLive kernel}\end{aligned}$ | We change only the self-attention kernel. | $\displaystyle \begin{aligned}&\mathrm{FPS}:1.30\to2.17\\[6pt]&\underbrace{24\text{–}27\ \mathrm{dB}}_{\text{PSNR of the first chunk to A′}}\end{aligned}$<br>[E29](../../notes/research/cards/E29.%20Вершина%20A′%20и%20спуск%20B,%20FP8,%20704×1280.md) |
| Not applied: gives no speed<br>$\displaystyle \mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}_{\rm Sage}$ | $\displaystyle \begin{aligned}&\text{Fused kernel with LongLive rounding.}\\[6pt]&\underbrace{0.029\%}_{\text{difference from the start per call}}\end{aligned}$ | We remove the separate RoPE pass. | $\displaystyle \begin{aligned}&\mathrm{FPS}:5.41\to5.45\\[6pt]&\text{The video diverges like another seed.}\end{aligned}$<br>[E30](../../notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md) |

$$
\underbrace{\mathrm{cfg}_{\rm start}=\mathrm{point\mbox{-}b2}}_{\text{W4A4, Sage 2, T5 BF16, KV FP8, }704\times1280},\qquad\mathrm{FPS}=5.41.
$$

[Starting point config](../../configs/experiments/director-original/point-b2-w4a4-sage-704x1280.json)

<!-- card mvp2-quantization -->
## [Quantization](../../docs/research/mvp2/Quantization.md)

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

Now consider what a rule does to the positions of the downscaled grid.

$$
\underbrace{0,\ 1,\ \ldots,\ h'-1}_{\text{positions of the downscaled grid}}
\ \xrightarrow{\ \cdot M_H\ }\
\underbrace{0,\ M_H,\ \ldots,\ M_H(h'-1)}_{\approx\,0\ldots h-1:\ \text{the range the model was trained on}},
\qquad
\underbrace{\varphi_j(\mathrm{pos})=\theta'_{H,j}\,\mathrm{pos}}_{\text{rotation angle of pair }j}
$$

PI multiplies the angle of every pair by one number; NTK and mix change only the low frequencies. We do not change the time axis. Overview of papers — [[Позиции RoPE|RoPE positions]].

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| Not applied: the best $M$ does not grow as the grid shrinks<br>$\displaystyle \theta_j\mapsto M^{\,j/20}\theta_j$<br>NTK<br>[YaRN, App. A.2](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\text{We keep the high frequencies and stretch the low ones.}\\[6pt]&\underbrace{M^\star(h')=\arg\min_M\mathcal L}_{\text{best }M\text{ at the step}}\end{aligned}$ | We check whether the best $M$ grows as the grid shrinks. | $\displaystyle \begin{aligned}&\underbrace{M^\star\approx1.2\text{–}1.3}_{672,\ 640,\ 608}\\[6pt]&\text{Almost constant, does not grow.}\\[6pt]&\underbrace{\overline{\mathcal L}_1=0.436}_{\text{held-out scenes, }M=1.2}\end{aligned}$<br>[Descent 09.11](file:///D:/Experiments/DIT/outputs/res-descent-2026-11-09/REPORT.md) |
| Not applied: no stable improvement<br>$\displaystyle \mathrm{shift}\mapsto\mathrm{scaleArea}\cdot\mathrm{shift}$<br>[Scaling Rectified Flow, (23)](https://arxiv.org/html/2403.03206#S5.SS3.SSS2) | $\displaystyle \begin{aligned}&\text{We change the noise schedule for a smaller frame.}\\[6pt]&576\times1056:\quad 5\mapsto4.11\mapsto3.63\end{aligned}$ | Fewer tokens, smaller shift. | $\displaystyle \begin{aligned}&\text{There is no stable improvement.}\\[6pt]&\mathrm{shift}=5\end{aligned}$ |
| Not applied: fitted to the training scenes<br>$\displaystyle (M_H,M_W,\mu)\mapsto\arg\min\overline{\mathcal L}_1$<br>CMA-ES | $\displaystyle \begin{aligned}&\text{We search for the profile on the training scenes and check on the held-out ones.}\\[6pt]&\underbrace{\text{cup, fox}}_{\text{training}},\quad\underbrace{\text{astronaut, two foxes}}_{\text{held-out}}\\[6pt]&\underbrace{144\ \text{points}}_{\text{seeds 0 and 1}},\quad\underbrace{\text{seed 2}}_{\text{held-out}}\end{aligned}$ | Is there a profile better than one number per axis. | $\displaystyle \begin{aligned}&(M_H,M_W,\mu)=(1.134,\ 1.104,\ 1)\\[6pt]&\underbrace{0.338}_{\text{training}},\quad\underbrace{0.382}_{\text{held-out}}\\[6pt]&\text{Plateau from the third generation.}\end{aligned}$<br>[Report, section 14](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle M_H=\frac{22}{h'},\qquad M_W=\frac{40}{w'}$<br>PI by the grid ratio<br>[Position Interpolation, §2.3](https://arxiv.org/html/2306.15595) | $\displaystyle \begin{aligned}&\text{We stretch the positions of the downscaled frame to the trained grid.}\\[6pt]&\mathrm{pos}\mapsto M_a\,\mathrm{pos},\qquad\underbrace{M_a(h'-1)\approx h-1}_{\text{the same range of positions}}\\[6pt]&\underbrace{608\times1120:\ M_H=\tfrac{22}{19},\ M_W=\tfrac{40}{35}}_{\text{check on held-out scenes}}\end{aligned}$ | No tuning: the multiplier is set by the grids. The object keeps its position and scale in the frame. | $\displaystyle \begin{aligned}&\underbrace{0.387}_{\text{held-out, }608}\ \text{versus}\ \underbrace{0.382}_{\rm CMA}\\[6pt]&\underbrace{0.442}_{\text{PI}},\ \underbrace{0.619}_{\text{no interpolation}},\ \underbrace{0.607}_{\text{uniform }1.1}\quad(512,\ \text{seed }0)\\[6pt]&\text{User: the objects stand where they are in the reference.}\end{aligned}$ |
| Not applied: $-6\%$ FPS, the loss does not decide<br>$\displaystyle 17\times30\mapsto17\times31$<br>Uniform stretching | $\displaystyle \begin{aligned}&\underbrace{544\times960:\ M_H=1.294,\ M_W=1.333}_{\text{non-uniform}}\\[6pt]&\underbrace{544\times992:\ M_H=1.294,\ M_W=1.290}_{\text{as in the reference}}\end{aligned}$ | We check whether different stretching of the two axes spoils the loss. The hypothesis is neither confirmed nor rejected: the sign depends on the cross-attention kernel. | $\displaystyle \begin{aligned}&\underbrace{\overline{\mathcal L}_1:\ 0.426\ \text{vs}\ 0.414}_{544\times960\ \text{vs}\ 544\times992,\ \text{cross SDPA}}\\[5pt]&\underbrace{\overline{\mathcal L}_1:\ 0.426\ \text{vs}\ 0.436}_{\text{the same, cross Sage 2}}\\[5pt]&\mathrm{FPS}:-6\%\end{aligned}$<br>[Report, sections 27, 30](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |

**Resolution ladder**

PI by the grid ratio, $\Sigma_{\rm ordered}$, FPS of the last chunk of the fox with fresh noise, before Sage 2 in cross-attention.

| $H\times W$ | $h\times w$ | FPS, $F_{\rm win}=32$ | FPS, $F_{\rm win}=24$ | $\overline{\mathcal L}_1$ | $\overline{\mathcal L}_1/\mathcal L_{\rm ceil}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $704\times1280$ | $22\times40$ | $5.71$ | $6.58$ | $0$ | $0$ |
| $608\times1120$ | $19\times35$ | $8.51$ | $9.61$ | $0.360$ | $0.49$ |
| $576\times1056$ | $18\times33$ | $9.90$ | $11.14$ | $0.393$ | $0.53$ |
| $544\times992$ | $17\times31$ | $11.83$ | $13.20$ | $0.414$ | $0.56$ |
| $544\times960$ | $17\times30$ | $12.53$ | $13.94$ | $0.426$ | $0.58$ |
| $512\times928$ | $16\times29$ | $14.18$ | $15.73$ | $0.483$ | $0.66$ |

The step $544\mapsto512$ costs $0.057$ of loss from $544\times960$ ($0.069$ from $544\times992$); the previous steps cost $0.02$–$0.03$.

[Grids](file:///D:/Experiments/DIT/outputs/speed-2026-11-12/grids/) · [NTK hypothesis](../../notes/research/NTK-zoom-hypothesis.md) · [Frequency-aware](../../notes/research/Frequency-aware-RoPE.md)

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

Now consider the window that self-attention reads, in latent frames.

$$
\mathrm{KVvid}_n\oplus\mathrm{KVnew}_{n+1}=\Bigl[\ \underbrace{\mathrm{KVanch}}_{F_{\rm anch}=8}\ \Big\vert\ \underbrace{\mathrm{KVupd}_n}_{\text{ring}}\ \Big\vert\ \underbrace{\mathrm{KVnew}_{n+1}}_{F_{\rm chunk}=8}\ \Bigr],
\qquad
F_{\rm win}=\begin{cases}32=8+16+8,&\text{MVP1}\\24=8+8+8\\16=8+0+8\end{cases}
$$

MVP1 memory rules — [[Длительная генерация|Long generation]].

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| Not applied: fails at a scene change, drift in 3 of 12 runs<br>$\displaystyle F_{\rm win}:32\mapsto24$ | $\displaystyle \begin{aligned}&\text{The ring is shorter by one chunk.}\\[6pt]&\underbrace{F_{\rm win}-F_{\rm anch}-F_{\rm chunk}:\ 16\mapsto8}_{\text{frames in the ring}}\\[6pt]&\text{The window takes effect when the ring is full.}\end{aligned}$ | Fewer keys in self-attention. | $\displaystyle \begin{aligned}&\mathrm{FPS}:+12\text{–}15\%\\[5pt]&\text{The first three chunks are bit-for-bit equal.}\\[5pt]&\underbrace{\lvert\Delta B\rvert>15}_{\text{blue shift, 12 chunks}}:\ \underbrace{3\ \text{of}\ 12}_{F_{\rm win}=24},\quad\underbrace{0\ \text{of}\ 12}_{F_{\rm win}=32}\\[5pt]&\underbrace{608\ (s=0),\quad576\ (s=1),\quad512\ (s=0)}_{\text{the three cases; none at }544\times960}\\[5pt]&\text{Scene change: the ring equals one chunk (cache.rs:135).}\end{aligned}$<br>[Report, section 27](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Not applied: the scene breaks, the seams are three times larger<br>$\displaystyle F_{\rm win}:32\mapsto16$ | $\displaystyle \begin{aligned}&\text{No ring: anchor and new chunk.}\\[6pt]&16=8+0+8\end{aligned}$ | The limit of shrinking the window. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:+27\%}_{608\times1120}\\[6pt]&\underbrace{3.86\ /\ 4.68}_{\text{seam: cup / fox}}\\[6pt]&\text{The cup empties, the frame resets.}\end{aligned}$ |
| Not applied: the video memory breaks<br>$\displaystyle \mathrm{KVnew}_{n+1}:\ t=0\mapsto t_2$<br>`cache_write: last_step` | $\displaystyle \begin{aligned}&\text{We take K/V from the last denoising step.}\\[6pt]&T_{\rm chunk}:\ 3\cdot30\,T_{\rm block}\mapsto2\cdot30\,T_{\rm block}\\[6pt]&\underbrace{\cos\bigl(V^{t_2},V^{t=0}\bigr)}_{\text{similarity to the clean write, per block}}\end{aligned}$ | We remove a third of the chunk. | $\displaystyle \begin{aligned}&\underbrace{\cos V=0.43,\quad\cos K=0.76}_{t_2=833}\\[6pt]&\underbrace{\mathrm{FPS}:8.60\to11.25}_{608\times1120}\\[6pt]&\text{Water in the cup does not fill;}\\[6pt]&\text{the fox's camera drifts away.}\end{aligned}$<br>[Report, part 3](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Not applied: $\cos V\le0.70$, below the $0.9$ threshold<br>$\displaystyle t_2:833\mapsto t$<br>Write at low noise | $\displaystyle \begin{aligned}&\text{We add noise to the clean chunk latent up to }t\text{ and compare K/V.}\\[6pt]&t\in\{600,400,250,100\}\end{aligned}$ | Is there a step time at which K/V are close to the clean ones. | $\displaystyle \begin{aligned}&\cos V:\quad0.48,\ 0.58,\ 0.66,\ 0.70\\[6pt]&\text{K/V depend on the modulation by }t\text{.}\end{aligned}$ |
| Not applied: $-1.4\%$ FPS<br>$\displaystyle \mathrm{KVvid}_n\mapsto\mathrm{KVanch}$<br>in the write pass<br>`clean_pass_sink` | $\displaystyle \begin{aligned}&\text{In the write pass, attention only on the anchor and the chunk.}\\[6pt]&\underbrace{T_{\rm Sage}:12.0\to4.1\ \mathrm{ms}}_{\text{estimate from the first chunk}}\end{aligned}$ | We make the third pass cheaper. | $\displaystyle \begin{aligned}&\mathrm{FPS}:-1.4\%\\[6pt]&\text{The mask adds its own overhead.}\end{aligned}$ |
| Measurement<br>$\displaystyle T_{\rm chunk}(\lvert\mathrm{KVvid}_n\rvert)$ | $\displaystyle \begin{aligned}&\text{Consider the chunk time while the window fills.}\\[5pt]&\underbrace{\lvert\mathrm{KVvid}_n\rvert=4080,\ 8160,\ 12240,\ 16320}_{\text{keys, chunks }0\ldots3,\ 544\times960}\\[5pt]&\underbrace{\Delta T_{\rm chunk}=90\cdot(2.70+0.34)\ \mathrm{ms}=0.274\ \mathrm{s}}_{\text{Sage self + its preparation, per chunk of window}}\end{aligned}$ | Sage is linear in the number of keys, so the window is the only cause of the growth of the chunk time. | $\displaystyle \begin{aligned}&\underbrace{T_{\rm chunk}:1.85\to2.03\to2.28\to2.57\ \mathrm{s}}_{\text{80 W}}\\[5pt]&\underbrace{+0.254,\ +0.288\ \mathrm{s}}_{\text{measured vs }0.274\text{ by the model}}\end{aligned}$<br>[Report 16.11](../../docs/report-2026-11-16-bottlenecks.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \Phi_{\rm continue}^{(2)}=\Phi_{\rm continue}^{(1)}$ | $\displaystyle \begin{aligned}&(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)\\[6pt]&\text{A separate write pass at }t=0\text{.}\end{aligned}$ | We keep the MVP1 video memory unchanged. | $\displaystyle \begin{aligned}&\underbrace{0\ \text{of}\ 12}_{\text{strong color drift}}\\[6pt]&\underbrace{1.71\ /\ 1.13}_{\text{seam: fox / cup, }512}\end{aligned}$ |

[Window grids](file:///D:/Experiments/DIT/outputs/kv-window-2026-11-11/) · [Report 11–12.11](../../docs/report-2026-11-11-loss-704-608-for-fable.md)

<!-- card mvp2-speed -->
# Compilation and placement · MVP2

What limits the chunk speed and what speeds up without changing the result? Consider the chunk time and the composition of one block.

$$
\underbrace{T_{\rm chunk}=3\cdot30\cdot T_{\rm block}+T_{\rm dec}}_{\text{three passes and the decoder}},\qquad
\underbrace{T_{\rm block}=T_{\rm GEMM}+T_{\rm Sage}+T_{\rm prep}+T_{\rm norm}+\cdots}_{\text{kernels of one block}},\qquad
\underbrace{\mathrm{FPS}=\frac{4\,F_{\rm chunk}}{T_{\rm chunk}}}_{\text{4 frames per latent frame}}
$$

Now consider where the time of one chunk goes and what bounds each part.

$$
\underbrace{T_{\rm chunk}=2.564\ \mathrm{s}}_{544\times960,\ F_{\rm win}=32,\ 80\ \mathrm{W}}:\qquad
\begin{array}{l|c|l}
\text{part}&\text{share}&\text{bound by}\\\hline
\text{Sage self}&38.0\%&\text{compute: tensor pipe }74\%\\
\text{GEMM W4A4}&36.2\%&\text{compute: 255 registers, occupancy }16.5\%\\
\text{norms, modulation}&6.9\%&\text{memory: DRAM }90\text{–}93\%\\
\text{Sage preparation}&5.5\%&\text{memory}\\
\text{decoder}&3.7\%&\text{compute}\\
\text{activation quantization}&2.7\%&\text{memory}\\
\text{cross, RoPE, KV write, idle}&7.0\%&
\end{array}
$$

Graph and fusions of MVP1 — [Compilation · MVP1](../../MVP1%20EN.canvas). Placement — [[Offloading]].

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{Attn}_{\rm cross}:\ \mathrm{SDPA}\mapsto\mathrm{Sage\,2}$ | $\displaystyle \begin{aligned}&\text{Consider the cross-attention of the block.}\\[6pt]&\underbrace{\mathrm{KVtext}}_{\text{computed once per prompt}}\end{aligned}$ | The same kernel as in self-attention. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:14.18\to14.71}_{512\times928,\ +3.7\%}\\[6pt]&\Delta\overline{\mathcal L}_1=+0.001\end{aligned}$<br>[Report, part 3](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \Sigma_{\rm ordered}\mapsto\Sigma_{\rm atomic}$<br>Live only | $\displaystyle \begin{aligned}&\text{In Live a bit-exact repeat is not needed.}\\[6pt]&B_{\rm VRAM}^{\rm margin}:0.0625\mapsto0\end{aligned}$ | We return the fast summation order and give back the VRAM margin. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:14.71\to14.82\to15.05}_{512\times928}\\[6pt]&\underbrace{13.44}_{544\times960}\end{aligned}$ |
| Not applied: $+2.8\%$ FPS for $\Delta\overline{\mathcal L}=+0.023$<br>$\displaystyle \mathrm{Attn}_{\rm self}\mapsto\mathrm{Attn}_{\rm sparse}$<br>`kernels.sparse` | $\displaystyle \begin{aligned}&\text{We drop the keys with a small contribution.}\\[6pt]&\underbrace{\varepsilon=0.02,\ \text{anchor}+8}_{\text{variant 1}},\qquad\underbrace{\text{anchor}+\text{last 4}}_{\text{variant 2}}\end{aligned}$ | Fewer keys in the Sage kernel. | $\displaystyle \begin{aligned}&\underbrace{+2.8\%,\quad\Delta\overline{\mathcal L}=+0.023}_{\text{variant 1}}\\[6pt]&\underbrace{+1.0\%}_{\text{variant 2}}\end{aligned}$ |
| Not applied: the GPU is already busy computing<br>$\displaystyle T_{\rm dec}\parallel T_{\rm block}$ | $\displaystyle \begin{aligned}&\text{The decoder on its own stream on top of the next chunk.}\\[6pt]&T_{\rm dec}=8\cdot12.7\ \mathrm{ms}\end{aligned}$ | We overlap the decoder with the blocks. | $\displaystyle \begin{aligned}&\text{The latent is bit-for-bit equal.}\\[6pt]&\mathrm{FPS}:13.44\to13.37\\[6pt]&M_{\rm VRAM}=8.07\ \text{GB}\end{aligned}$<br>[Report, section 30](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Not applied: impossible bit-for-bit<br>$\displaystyle \mathrm{prep}(K_{\rm win}):\ 3\mapsto1$<br>per chunk | $\displaystyle \begin{aligned}&\text{The window of past chunks does not change between passes.}\\[6pt]&\underbrace{\overline K,\ \operatorname{scale}(V)}_{\text{over all keys, including the new chunk}}\end{aligned}$ | Do not recompute the preparation of the window keys. | $\displaystyle \begin{aligned}&\underbrace{2.2\ \text{GB}}_{\text{cache of the prepared window in VRAM}}\\[6pt]&\underbrace{\le3.6\%}_{\text{estimate}}\end{aligned}$ |
| Not applied: ceiling $+0.4\%$<br>$\displaystyle \mathrm{Norm}\to\mathrm{Quant}_4\mapsto\mathrm{fused}$ | $\displaystyle \begin{aligned}&\text{Was: the norm writes }y\text{ in BF16, the quantizer reads it again.}\\[5pt]&\text{Fused bit-for-bit, the quantizer must read }X\text{ and the attention output.}\\[5pt]&\underbrace{75\to50\ \mathrm{MB}}_{\text{QKV input}},\qquad\underbrace{125\to125\ \mathrm{MB}}_{\text{cross Q and FFN inputs}}\end{aligned}$ | We remove one write and one read of BF16. The compiler already fuses the norm with the residual, so only the QKV input gains. | $\displaystyle \begin{aligned}&\underbrace{10\ \mathrm{ms}\ \text{of}\ 2564}_{\text{per chunk, count of bytes}}\end{aligned}$<br>[Report 16.11](../../docs/report-2026-11-16-bottlenecks.md) |
| Not applied: $-1.2\%$ under the power limit<br>$\displaystyle \mathrm{tile}:\ 256\times128\mapsto256\times64$<br>GEMM W4A4 | $\displaystyle \begin{aligned}&\text{Consider one thread block of the INT4 GEMM.}\\[5pt]&\text{Was: 254 registers per thread, one block per SM.}\\[5pt]&\text{Now: half the tile in }N\text{, 125 registers, two blocks per SM.}\end{aligned}$ | A denser kernel needs fewer cycles, but at the power limit the driver lowers the clock by the same amount. | $\displaystyle \begin{aligned}&\underbrace{16.6\to32.7\%}_{\text{occupancy}},\quad\underbrace{63\to70\%}_{\text{tensor pipe}}\\[5pt]&\underbrace{-8\ldots10\%}_{\text{cycles}},\quad\underbrace{-9\ldots15\%}_{\text{SM clock}}\\[5pt]&\underbrace{-1.2\%}_{\text{chunk, at the limit}},\quad\underbrace{+3.4\%}_{\text{if the clock held}}\end{aligned}$<br>[Report 16.11](../../docs/report-2026-11-16-bottlenecks.md) |
| Not applied: $+0.8\%$ for 70 MB of VRAM<br>$\displaystyle \operatorname{sum,max}(K_{\rm win},V_{\rm win}):\ 3\mapsto1$<br>per chunk | $\displaystyle \begin{aligned}&\text{Consider the partial sums of }K\text{ and maxima of }V\text{ over the tiles of the window.}\\[5pt]&\text{They are equal in the three passes; we keep them from the first one.}\end{aligned}$ | The same partial sums in the same order, so the result is bit-for-bit equal. | $\displaystyle \begin{aligned}&\underbrace{0.57\to0.22\ \mathrm{ms}}_{\text{per call, passes 2 and 3}}\\[5pt]&\underbrace{21\ \mathrm{ms}\ \text{of}\ 2564}_{\text{per chunk, estimate}}\end{aligned}$<br>[Report 16.11](../../docs/report-2026-11-16-bottlenecks.md) |
| Measurement<br>$\displaystyle T_{\rm idle}$ | $\displaystyle \begin{aligned}&\text{Consider the time when the GPU queue is empty.}\\[5pt]&\underbrace{18}_{\text{gaps between kernels}}+\underbrace{22}_{\text{frames to ffmpeg, offline only}}+\underbrace{5}_{\text{noise on CPU}}+\underbrace{3}_{\text{decoder, host}}\ \mathrm{ms}\end{aligned}$ | The host does not hold the GPU back: in Live only about 7 ms can be removed. | $\displaystyle \begin{aligned}&\underbrace{48.5\ \mathrm{ms}=1.9\%}_{\text{offline}},\qquad\underbrace{\approx26\ \mathrm{ms}}_{\text{Live}}\end{aligned}$<br>[Report 16.11](../../docs/report-2026-11-16-bottlenecks.md) |
| Open: set by the user<br>$\displaystyle P_{\rm limit}:\ 100\mapsto80\ \mathrm{W}$ | $\displaystyle \begin{aligned}&\text{Consider the power limit of the GPU driver.}\\[5pt]&\underbrace{\text{sw\_power\_cap}:\ 89\%\mapsto96\%}_{\text{share of samples at the limit}}\\[5pt]&\underbrace{f_{\rm SM}:2322\to2128\ \mathrm{MHz}}_{\text{clock under load}}\end{aligned}$ | Both compute kernels (74% of the chunk) run at the clock that the power limit allows. This is the only large lever that does not change the math. | $\displaystyle \begin{aligned}&\underbrace{T_{\rm chunk}:2.405\to2.564\ \mathrm{s}}_{-6.6\%}\\[5pt]&\underbrace{105\ \mathrm{W}}_{\text{vBIOS maximum, not measured}}\end{aligned}$<br>[Report 16.11](../../docs/report-2026-11-16-bottlenecks.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathrm{policy}_2=\mathrm{greedy\_vram}$ | $\displaystyle \begin{aligned}&\text{Consider the delivery of the weight rows.}\\[6pt]&\underbrace{7.1\ \mathrm{ms}}_{\text{row copy RAM → VRAM}}<\underbrace{T_{\rm block}}_{\text{hidden behind the computation}}\end{aligned}$ | We check whether the weight transfer is the limit. | $\displaystyle \begin{aligned}&\underbrace{1\ \mathrm{ms}}_{\text{waiting for copies per run}},\quad\underbrace{0}_{\text{late arrivals}}\\[6pt]&\text{The speed is limited by the block computation.}\end{aligned}$<br>[Report, section 12](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |

$$
\underbrace{T_{\rm chunk}=2.409\ \mathrm{s}}_{100\ \mathrm{W}}
=\underbrace{90\cdot25.47\ \mathrm{ms}}_{\text{blocks, }95\%}+\underbrace{0.12\ \mathrm{s}}_{\text{decoder, gaps}},\qquad
\mathrm{FPS}=\frac{32}{2.409}\approx13.3,\qquad
\underbrace{\mathrm{FPS}\approx12.5}_{80\ \mathrm{W}}.
$$

Transformations with a bit-for-bit equal latent are exhausted: each gives less than $1\%$ under the power limit. To $15$ FPS we lack $11.6\%$ at 100 W and $20\%$ at 80 W.

[Bottleneck report 16.11](../../docs/report-2026-11-16-bottlenecks.md)

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

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| Measurement<br>$\displaystyle \mathrm{video}_0\mapsto\operatorname{resize}_{608}(\mathrm{video}_0)$ | $\displaystyle \begin{aligned}&\text{Downscale the finished reference by area, without generation.}\\[6pt]&\mathcal L\bigl(\operatorname{resize}_{608}(\mathrm{video}_0),\mathrm{video}_0\bigr)\end{aligned}$ | The price of the pixel loss itself. | $\displaystyle \begin{aligned}&\underbrace{0.013}_{\text{fox}},\quad\underbrace{0.008}_{\text{cup}}\\[6pt]&\text{3 chunks, seed 0}\end{aligned}$ |
| Measurement<br>$\displaystyle \mathrm{cfg}_0\mapsto\mathrm{cfg}_0$<br>Repeat without changes. | $\displaystyle \begin{aligned}&\text{Consider float summation from parallel kernel tiles.}\\[6pt]&\underbrace{\Sigma_{\rm atomic}}_{\text{order of summands is not fixed}}\\[6pt]&\underbrace{\overline K}_{\text{mean of the keys before Sage}},\qquad\underbrace{x\,W_{\rm down}}_{\text{low-rank branch of W4A4}}\end{aligned}$ | We check whether one config gives one clip. Two sources: our key preparation kernel and the Nunchaku kernel. | $\displaystyle \begin{aligned}&\underbrace{36\text{–}38\%}_{\text{divergence of the first chunk latent}}\\[6pt]&\mathcal L_{\rm floor}:\quad\underbrace{0.26}_{\text{fox}},\quad\underbrace{0.18}_{\text{cup}}\\[6pt]&\text{The divergence starts at block 0.}\end{aligned}$<br>[Report, section 3](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \Sigma_{\rm atomic}\mapsto\Sigma_{\rm ordered}$<br>`kernels.lora_sum` | $\displaystyle \begin{aligned}&\text{Each tile writes its own partial sum.}\\[6pt]&\underbrace{p_1,\ldots,p_m}_{\text{partial sums of the tiles}}\\[6pt]&\Sigma_{\rm ordered}=\bigl((p_1+p_2)+\cdots\bigr)+p_m\\[6pt]&\text{We sum in a fixed order.}\end{aligned}$ | One config gives one clip: the repeat is comparable bit by bit. In experiments ordered, in Live atomic. | $\displaystyle \begin{aligned}&\mathcal L_{\rm floor}=0\\[6pt]&\text{The latent and MP4 of two repeats are bit-for-bit equal.}\\[6pt]&\underbrace{T_{\rm block}:38.0\to38.8\ \mathrm{ms}}_{608\times1120},\quad\mathrm{FPS}:-2\%\end{aligned}$<br>[Report, section 11](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Measurement<br>$\displaystyle M:1\mapsto1+10^{-5}$<br>`rope.m` | $\displaystyle \begin{aligned}&\text{One small perturbation, three variants of weights and attention.}\\[6pt]&\underbrace{\delta=\frac{\lVert\mathrm{lat}'-\mathrm{lat}\rVert}{\lVert\mathrm{lat}\rVert}}_{\text{divergence of the first chunk latent}}\\[6pt]&704\times1280,\quad\text{fox},\quad\Sigma_{\rm ordered}\end{aligned}$ | Where the amplification of a small error comes from: the two-step model or the 4 bits. | $\displaystyle \begin{aligned}&\underbrace{\delta=0.35}_{\mathrm{W4A4}+\mathrm{Sage\,2}}\\[6pt]&\underbrace{\delta=0.18}_{\mathrm{FP8}+\mathrm{Sage\,2}}\\[6pt]&\underbrace{\delta=0.16}_{\mathrm{FP8}+\mathrm{SDPA}}\\[6pt]&\text{The model itself amplifies; W4A4 doubles it.}\end{aligned}$ |
| <span class="wan-kept">Applied</span><br>$\displaystyle \mathcal L\mapsto\overline{\mathcal L}_1$ | $\displaystyle \begin{aligned}&\text{We average the loss of the first chunk.}\\[6pt]&\underbrace{s\in\{0,1,2\}}_{\text{seed}},\quad\underbrace{4\ \text{scenes}}_{\text{cup, fox, astronaut, two foxes}}\\[6pt]&\text{Order of the chains as in the reference:}\\[6pt]&\text{the seed noise runs consecutively through the chains.}\end{aligned}$ | The loss accumulates already in the first chunk; one point costs about 5 s. Averaging removes the dependence on a single trajectory. | $\displaystyle \begin{aligned}&\underbrace{\overline{\mathcal L}_1=0.360,\quad\overline{\mathcal L}=0.398}_{608\times1120}\\[6pt]&\underbrace{\mathcal L_{\rm ceil}=0.736}_{\text{first chunk}},\quad\underbrace{0.753}_{\text{clip}}\end{aligned}$<br>[Report, section 27](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle \epsilon\mapsto\operatorname{spectral}_{h\times w}(\epsilon_{22\times40})$<br>`descent.noise` | $\displaystyle \begin{aligned}&\text{We get the noise of the downscaled frame from the noise of the reference.}\\[6pt]&\underbrace{\epsilon_{22\times40}}_{\text{reference noise}}\xrightarrow{\text{truncate the spectrum}}\underbrace{\epsilon_{h\times w}}_{\text{noise of the downscaled frame}}\\[6pt]&\text{Computed on the CPU at every chunk.}\end{aligned}$ | The reference and the downscaled frame share the low frequencies of the noise. For comparison only; in Live the noise is fresh. | $\displaystyle \begin{aligned}&\underbrace{0.43\ \mathrm{s}}_{\text{per chunk}}\\[6pt]&\underbrace{\mathrm{FPS}:8.60\to9.73}_{608\times1120,\ \text{fresh noise}}\\[6pt]&\text{We measure FPS with fresh noise,}\\[6pt]&\text{the loss with the spectral one.}\end{aligned}$ |

[Report 11–12.11](../../docs/report-2026-11-11-loss-704-608-for-fable.md) · [Measurement plan](../../docs/handoff-2026-11-11-loss-search.md)

<!-- card mvp2-boundary -->
---

**Choice of the working point 544×960, window 32 · next stage**

<!-- card mvp2-result -->
# Working point · MVP2

Which configuration do we fix? Consider the candidates near the constraint $\mathrm{FPS}\ge15$.

$$
\min_{\mathrm{cfg}}\ \overline{\mathcal L}_1(\mathrm{cfg})\quad\text{subject to}\quad\mathrm{FPS}\ge15,\quad M_{\rm VRAM}\le8\ \text{GB},\quad M_{\rm RAM}\le16\ \text{GB}.
$$

| Candidate | FPS, Live | $M_{\rm VRAM}$ | $\overline{\mathcal L}_1$ | $\overline{\mathcal L}_1/\mathcal L_{\rm ceil}$ | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <span class="wan-kept">MVP2 result</span><br>$544\times960,\ F_{\rm win}=32$ | $13.44$ | $7.3$ GB | $0.426$ | $0.58$ | User's choice by the grids: quality matters more than $1.5$ FPS |
| $544\times960,\ F_{\rm win}=24$ | $15.03$ | $6.8$ GB | $0.426$ | $0.58$ | Not applied: fails at a scene change |
| $512\times928,\ F_{\rm win}=32$ | $15.05$ | $7.1$ GB | $0.484$ | $0.66$ | Fallback point at 15 FPS |

$$
\begin{aligned}
\mathrm{cfg}_{\rm MVP2}&:\quad H\times W=544\times960,\quad h\times w=17\times30,\\[6pt]
&\underbrace{M_H=\tfrac{22}{17},\quad M_W=\tfrac{40}{30}}_{\text{PI by the grid ratio}},\quad
\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)}_{\text{MVP1 video memory}},\\[6pt]
&\underbrace{W_{\rm DiT}=\mathrm{W4A4},\quad W_{\rm T5}=\mathrm{BF16},\quad\mathrm{KV}=\mathrm{FP8}}_{\text{precision}},\quad
\underbrace{\mathrm{Attn}_{\rm self}=\mathrm{Attn}_{\rm cross}=\mathrm{Sage\,2}}_{\text{attention kernels}},\\[6pt]
&\underbrace{\mathrm{FPS}=13.44<15}_{\text{constraint not met}},\quad\overline{\mathcal L}_1=0.426.
\end{aligned}
$$

The constraint $\mathrm{FPS}\ge15$ is not met at the chosen point. Status 16.11: $13.3$ FPS at 100 W and $12.5$ at 80 W in the steady state. The transformations without a change of the math are measured and give less than $1\%$ each; the open lever is the power limit. We do not change the resolution and the window. [Bottleneck report](../../docs/report-2026-11-16-bottlenecks.md)

[Live config](../../configs/candidates/544x960-win32-live.json) · [Experiment config](../../configs/candidates/544x960-win32-experiment.json) · [512×928](../../configs/candidates/512x928-win32-live.json)

[Final grids](file:///D:/Experiments/DIT/outputs/speed-2026-11-12/grids/) · [Report 11–12.11](../../docs/report-2026-11-11-loss-704-608-for-fable.md) · [Version summary](../../Версии%20исследования%20EN.canvas)
