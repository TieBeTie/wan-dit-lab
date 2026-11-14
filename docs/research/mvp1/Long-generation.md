<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Long generation

**Before choosing LongLive · Wan / Self-Forcing**

How can we continue a video under limited memory? Consider the KV of one block.

$$
\underbrace{n}_{\text{number of finished chunks}},\quad\underbrace{\mathrm{lat}_n}_{\text{finished chunk}},\quad
\underbrace{\mathrm{KVvid}_n=(K_n,V_n)}_{\text{stored video memory}},\quad
\underbrace{\mathrm{ctx}}_{\text{unchanged text condition}}
$$
$$
(\mathrm{KVvid}_n,\mathrm{ctx},\underbrace{\epsilon_{n+1}}_{\text{new noise}})
\xrightarrow{\text{generation of the next chunk}}
(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1}),\qquad n=0,1,\ldots
$$

Now consider the structure of the memory: which K/V we keep for the next step. We split it into a permanent anchor and an updated part.

$$
\underbrace{\oplus}_{\text{concatenation of K and V rows}},\qquad
\underbrace{\mathrm{KVvid}_n=\mathrm{KVanch}\oplus\mathrm{KVupd}_n}_{\text{memory before generating chunk }n+1},\qquad
\underbrace{\mathrm{KVanch}}_{\text{K/V of the initial frames}},\quad
\underbrace{\mathrm{KVupd}_n}_{\text{updated part of K/V}}
$$
$$
\underbrace{\mathrm{KVnew}_{n+1}=(K_{\rm new},V_{\rm new})}_{\text{K/V of the new chunk}},\qquad

\underbrace{N_{\rm max}}_{\text{maximum number of tokens in attention}}
$$

	Φ denotes the memory update rule; n → n+1 means the transition to the next chunk.

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\Phi_{\rm sink}\colon\\[4pt]\mathrm{KVvid}_n\mapsto\mathrm{KVanch}\oplus\mathrm{KVupd}_n\end{gathered}$<br>[Deep Sink: Deep Forcing, §4.2](https://arxiv.org/html/2512.05081v1#S4.SS2) | $\displaystyle \begin{aligned}&\text{We keep a large anchor: about half of the window.}\\[6pt]&\underbrace{F_{\rm win}=21}_{\text{frames in the window}},\quad\underbrace{F_{\rm anch}=10}_{\text{anchor frames}},\quad\underbrace{F_{\rm chunk}=3}_{\text{frames of the new chunk}}\\[8pt]&\underbrace{21=10+8+3}_{\text{anchor + changeable memory + new chunk}}\\[8pt]&\underbrace{N_{\rm frame}}_{\text{tokens of one latent frame}},\quad N_{\rm max}=F_{\rm win}N_{\rm frame},\quad\lvert\mathrm{KVanch}\rvert=F_{\rm anch}N_{\rm frame}\\[6pt]&\text{All frame counts here refer to latents, not to seconds.}\end{aligned}$ | A large initial anchor keeps the support on the original scene. | $\displaystyle \begin{aligned}&\text{We tuned the anchor size }F_{\rm anch}.\\[6pt]&\underbrace{F_{\rm anch}\in\{0,3,6,9\}}_{\text{first in steps of one chunk}}\\[8pt]&F_{\rm anch}=9:\ \text{liked},\\[2pt]&\text{drift remains}.\\[6pt]&\underbrace{F_{\rm anch}\in\{12,15,18\}}_{\text{extended the search}}\\[8pt]&\text{Then we compared with Deep Forcing:}\\[5pt]&\underbrace{F_{\rm anch}\approx F_{\rm win}/2}_{\text{anchor about half of the window}}\\[8pt]&\underbrace{F_{\rm anch}=10}_{F_{\rm win}=21},\quad\underbrace{F_{\rm anch}=14}_{\text{authors' config}}\\[8pt]&\text{We compared 10 and 14;}\\[3pt]&\underbrace{F_{\rm anch}=10}_{\text{kept as the working variant}}\end{aligned}$<br><br>Anchor search · 10 vs 14 |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\Phi_{\rm time}\colon\\[4pt]k_i^{\rm stored}\mapsto\widetilde k_i\end{gathered}$<br>[Temporal RoPE: Deep Forcing, §4.2](https://arxiv.org/html/2512.05081v1#S4.SS2)<br>Our implementation: rotation of a copy. | $\displaystyle \begin{aligned}&\text{In a full window we limit the temporal distance to the anchor.}\\[6pt]&\underbrace{\mathrm{pos}_i^{\rm new}}_{\text{temporal position for attention}}:\quad\underbrace{0,\ldots,9}_{\text{anchor}},\quad\underbrace{10,\ldots,17}_{\text{ring}},\quad\underbrace{18,19,20}_{\text{new chunk}}\\[8pt]&\text{Consider one temporal coordinate pair of the 3D RoPE key.}\\[6pt]&\underbrace{\Delta_i=\mathrm{pos}_i^{\rm new}-\mathrm{pos}_i^{\rm write}}_{\text{shift relative to the position at write time}},\quad\underbrace{\omega_{\rm time}}_{\text{frequency of this pair}}\\[8pt]&\underbrace{\operatorname{Rot}(\varphi)=\begin{pmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{pmatrix}}_{\text{rotation of a coordinate pair}}\\[8pt]&\widetilde k_i^{\rm time}=\operatorname{Rot}(\omega_{\rm time}\Delta_i)\,k_i^{{\rm time},{\rm stored}}\\[6pt]&\widetilde k_i^{\rm spatial}=k_i^{{\rm spatial},{\rm stored}},\quad\widetilde v_i=v_i\\[6pt]&\text{This is how we rotate every temporal pair; the stored K is not changed.}\end{aligned}$ | We limit the temporal distance to the anchor. We rotate a copy of the key; the rounding of repeated rotations does not accumulate in the stored K. | $\displaystyle \begin{aligned}&\text{An addition to the permanent anchor:}\\[3pt]&\text{we keep the frames, we limit}\\[2pt]&\text{the temporal distance.}\\[8pt]&\underbrace{F_{\rm anch}=10}_{\text{anchor}}+\underbrace{\text{window}}_{\text{temporal positions}}\\[3pt]&\text{applied together}.\\[8pt]&\underbrace{14/14}_{\text{CPU tests of the KV cache}}\\[8pt]&\underbrace{F_{\rm anch}\in\{10,14\}}_{\text{positions checked directly}}\\[8pt]&\underbrace{\text{video / window}}_{\text{with the same anchor}}\\[3pt]&\text{there is no separate A/B}\end{aligned}$ |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\Phi_{\rm PC}\colon\\[4pt](\mathrm{KVanch},\mathrm{KVupd}_n,\mathrm{KVnew}_{n+1})\\[2pt]\mapsto(\mathrm{KVanch},\mathrm{KVupd}_{n+1})\end{gathered}$<br>Participative Compression<br>[PC: Deep Forcing, §4.3](https://arxiv.org/html/2512.05081v1#S4.SS3) | $\displaystyle \begin{aligned}&\text{Consider a full cache }\mathrm{KVvid}_n=\mathrm{KVanch}\oplus\mathrm{KVupd}_n.\\[5pt]&\text{We free space for }\mathrm{KVnew}_{n+1}\text{ only inside }\mathrm{KVupd}_n.\\[7pt]&\underbrace{N_{\rm new}=\lvert\mathrm{KVnew}_{n+1}\rvert}_{\text{number of new K/V row pairs}}\\[7pt]&\underbrace{\mathrm{KVupd}_n^{\rm keep}\subseteq\mathrm{KVupd}_n}_{\text{protected rows of the last finished frames}}\\[7pt]&\underbrace{\mathrm{IDX}_{n,\rm keep}=\mathrm{rows}(\mathrm{KVupd}_n)\setminus\mathrm{rows}(\mathrm{KVupd}_n^{\rm keep})}_{\text{indices of candidates inside }\mathrm{KVupd}_n}\\[8pt]&\text{At the first denoising step of the new chunk:}\\[5pt]&\underbrace{\mathrm{score}_i=\sum_{q=1}^{N_{\rm new}}\langle Q_q,\widetilde k_i\rangle}_{\text{link of row }i\text{ of }\mathrm{KVupd}_n\text{ with the new chunk}},\quad i\in \mathrm{IDX}_{n,\rm keep}\\[6pt]&\underbrace{Q_q}_{\text{new queries}},\quad\underbrace{\widetilde k_i}_{\text{key from }\mathrm{KVupd}_n\text{ after Temporal RoPE}}\\[7pt]&\underbrace{\mathrm{IDX}_{\rm evict}=\operatorname{Bottom}_{N_{\rm new}}(\mathrm{IDX}_{n,\rm keep};\mathrm{score})}_{\text{indices of the }N_{\rm new}\text{ smallest scores}}\\[8pt]&\underbrace{\mathrm{KVupd}_{n+1}[\mathrm{IDX}_{\rm evict}]\leftarrow\mathrm{KVnew}_{n+1}}_{\text{we replace K and V in the same rows}}\\[6pt]&\mathrm{KVupd}_{n+1}[i]=\mathrm{KVupd}_n[i],\quad i\notin \mathrm{IDX}_{\rm evict}\\[6pt]&\mathrm{KVvid}_{n+1}=\underbrace{\mathrm{KVanch}}_{\text{unchanged}}\oplus\mathrm{KVupd}_{n+1}.\end{aligned}$ | PC and FIFO update $\mathrm{KVupd}_n$; $\mathrm{KVanch}$ is kept. FIFO evicts the oldest rows of $\mathrm{KVupd}_n$, PC chooses the rows with the weakest link to the new chunk. Temporal RoPE is applied in both cases. | $\displaystyle \begin{aligned}&\underbrace{N_{\rm pairs}=16}_{\text{pairs}},\quad480\times832,\quad\mathrm{BF16}\\[6pt]&\underbrace{d_j=Q_j^{\rm PC}-Q_j^{\rm FIFO}}_{Q:\ \text{MUSIQ at 15…19 s}}\\[7pt]&\underbrace{5:11}_{\text{wins PC : FIFO}},\quad\overline d=-1.384\\[7pt]&H_0:\Pr(d_j>0\mid d_j\ne0)=1/2\\[5pt]&\underbrace{p_{\rm value}=0.210>0.05}_{\text{two-sided sign test}}\\[5pt]&H_0\text{ is not rejected}.\\[7pt]&\text{By frames: the result depends on the scene.}\end{aligned}$<br><br>480p: videos and results<br>Frames: island · turtle<br>480p config · Previous 256p series |



**Implementation: ring KV cache**

FIFO (First In, First Out) and PC change only $\mathrm{KVupd}_n$. The anchor $\mathrm{KVanch}$ stays the same; the buffer reuses the chosen places in $\mathrm{KVupd}_n$.

$$
\underbrace{\mathrm{IDX}_{\rm evict}}_{\text{indices of evicted rows}}:
\begin{cases}
\text{the oldest rows},&\mathrm{FIFO},\\
\text{the smallest scores of the link to the new chunk},&\mathrm{PC},
\end{cases}
\qquad
\mathrm{IDX}_{\rm evict}\subseteq\mathrm{rows}(\mathrm{KVupd}_n),\qquad
\mathrm{KVupd}_{n+1}[\mathrm{IDX}_{\rm evict}]\leftarrow\mathrm{KVnew}_{n+1},\qquad
\mathrm{KVvid}_{n+1}=\mathrm{KVanch}\oplus\mathrm{KVupd}_{n+1}.
$$

The ring reuses a fixed buffer; FIFO/PC choose the rows to be replaced.

Implementation: KVCache.commit, evicted, place, finished


Model choice — [[Model]].
