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
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\Phi\_{\rm sink}\colon\\\\\mathrm{KVvid}\_n\mapsto\mathrm{KVanch}\oplus\mathrm{KVupd}\_n\end{gathered}$<br>[Deep Sink: Deep Forcing, §4.2](https://arxiv.org/html/2512.05081v1#S4.SS2) | $\displaystyle \begin{aligned}&\text{We keep a large anchor: about half of the window.}\\\\&\underbrace{F\_{\rm win}=21}\_{\text{frames in the window}},\quad\underbrace{F\_{\rm anch}=10}\_{\text{anchor frames}},\quad\underbrace{F\_{\rm chunk}=3}\_{\text{frames of the new chunk}}\\\\&\underbrace{21=10+8+3}\_{\text{anchor + changeable memory + new chunk}}\\\\&\underbrace{N\_{\rm frame}}\_{\text{tokens of one latent frame}},\quad N\_{\rm max}=F\_{\rm win}N\_{\rm frame},\quad\lvert\mathrm{KVanch}\rvert=F\_{\rm anch}N\_{\rm frame}\\\\&\text{All frame counts here refer to latents, not to seconds.}\end{aligned}$ | A large initial anchor keeps the support on the original scene. | $\displaystyle \begin{aligned}&\text{We tuned the anchor size }F\_{\rm anch}.\\\\&\underbrace{F\_{\rm anch}\in\\{0,3,6,9\\}}\_{\text{first in steps of one chunk}}\\\\&F\_{\rm anch}=9:\ \text{liked},\\\\&\text{drift remains}.\\\\&\underbrace{F\_{\rm anch}\in\\{12,15,18\\}}\_{\text{extended the search}}\\\\&\text{Then we compared with Deep Forcing:}\\\\&\underbrace{F\_{\rm anch}\approx F\_{\rm win}/2}\_{\text{anchor about half of the window}}\\\\&\underbrace{F\_{\rm anch}=10}\_{F\_{\rm win}=21},\quad\underbrace{F\_{\rm anch}=14}\_{\text{authors' config}}\\\\&\text{We compared 10 and 14;}\\\\&\underbrace{F\_{\rm anch}=10}\_{\text{kept as the working variant}}\end{aligned}$<br><br>Anchor search · 10 vs 14 |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\Phi\_{\rm time}\colon\\\\k\_i^{\rm stored}\mapsto\widetilde k\_i\end{gathered}$<br>[Temporal RoPE: Deep Forcing, §4.2](https://arxiv.org/html/2512.05081v1#S4.SS2)<br>Our implementation: rotation of a copy. | $\displaystyle \begin{aligned}&\text{In a full window we limit the temporal distance to the anchor.}\\\\&\underbrace{\mathrm{pos}\_i^{\rm new}}\_{\text{temporal position for attention}}:\quad\underbrace{0,\ldots,9}\_{\text{anchor}},\quad\underbrace{10,\ldots,17}\_{\text{ring}},\quad\underbrace{18,19,20}\_{\text{new chunk}}\\\\&\text{Consider one temporal coordinate pair of the 3D RoPE key.}\\\\&\underbrace{\Delta\_i=\mathrm{pos}\_i^{\rm new}-\mathrm{pos}\_i^{\rm write}}\_{\text{shift relative to the position at write time}},\quad\underbrace{\omega\_{\rm time}}\_{\text{frequency of this pair}}\\\\&\underbrace{\operatorname{Rot}(\varphi)=\begin{pmatrix}\cos\varphi&-\sin\varphi\\\\\sin\varphi&\cos\varphi\end{pmatrix}}\_{\text{rotation of a coordinate pair}}\\\\&\widetilde k\_i^{\rm time}=\operatorname{Rot}(\omega\_{\rm time}\Delta\_i)\\,k\_i^{{\rm time},{\rm stored}}\\\\&\widetilde k\_i^{\rm spatial}=k\_i^{{\rm spatial},{\rm stored}},\quad\widetilde v\_i=v\_i\\\\&\text{This is how we rotate every temporal pair; the stored K is not changed.}\end{aligned}$ | We limit the temporal distance to the anchor. We rotate a copy of the key; the rounding of repeated rotations does not accumulate in the stored K. | $\displaystyle \begin{aligned}&\text{An addition to the permanent anchor:}\\\\&\text{we keep the frames, we limit}\\\\&\text{the temporal distance.}\\\\&\underbrace{F\_{\rm anch}=10}\_{\text{anchor}}+\underbrace{\text{window}}\_{\text{temporal positions}}\\\\&\text{applied together}.\\\\&\underbrace{14/14}\_{\text{CPU tests of the KV cache}}\\\\&\underbrace{F\_{\rm anch}\in\\{10,14\\}}\_{\text{positions checked directly}}\\\\&\underbrace{\text{video / window}}\_{\text{with the same anchor}}\\\\&\text{there is no separate A/B}\end{aligned}$ |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\Phi\_{\rm PC}\colon\\\\(\mathrm{KVanch},\mathrm{KVupd}\_n,\mathrm{KVnew}\_{n+1})\\\\\mapsto(\mathrm{KVanch},\mathrm{KVupd}\_{n+1})\end{gathered}$<br>Participative Compression<br>[PC: Deep Forcing, §4.3](https://arxiv.org/html/2512.05081v1#S4.SS3) | $\displaystyle \begin{aligned}&\text{Consider a full cache }\mathrm{KVvid}\_n=\mathrm{KVanch}\oplus\mathrm{KVupd}\_n.\\\\&\text{We free space for }\mathrm{KVnew}\_{n+1}\text{ only inside }\mathrm{KVupd}\_n.\\\\&\underbrace{N\_{\rm new}=\lvert\mathrm{KVnew}\_{n+1}\rvert}\_{\text{number of new K/V row pairs}}\\\\&\underbrace{\mathrm{KVupd}\_n^{\rm keep}\subseteq\mathrm{KVupd}\_n}\_{\text{protected rows of the last finished frames}}\\\\&\underbrace{\mathrm{IDX}\_{n,\rm keep}=\mathrm{rows}(\mathrm{KVupd}\_n)\setminus\mathrm{rows}(\mathrm{KVupd}\_n^{\rm keep})}\_{\text{indices of candidates inside }\mathrm{KVupd}\_n}\\\\&\text{At the first denoising step of the new chunk:}\\\\&\underbrace{\mathrm{score}\_i=\sum\_{q=1}^{N\_{\rm new}}\langle Q\_q,\widetilde k\_i\rangle}\_{\text{link of row }i\text{ of }\mathrm{KVupd}\_n\text{ with the new chunk}},\quad i\in \mathrm{IDX}\_{n,\rm keep}\\\\&\underbrace{Q\_q}\_{\text{new queries}},\quad\underbrace{\widetilde k\_i}\_{\text{key from }\mathrm{KVupd}\_n\text{ after Temporal RoPE}}\\\\&\underbrace{\mathrm{IDX}\_{\rm evict}=\operatorname{Bottom}\_{N\_{\rm new}}(\mathrm{IDX}\_{n,\rm keep};\mathrm{score})}\_{\text{indices of the }N\_{\rm new}\text{ smallest scores}}\\\\&\underbrace{\mathrm{KVupd}\_{n+1}[\mathrm{IDX}\_{\rm evict}]\leftarrow\mathrm{KVnew}\_{n+1}}\_{\text{we replace K and V in the same rows}}\\\\&\mathrm{KVupd}\_{n+1}[i]=\mathrm{KVupd}\_n[i],\quad i\notin \mathrm{IDX}\_{\rm evict}\\\\&\mathrm{KVvid}\_{n+1}=\underbrace{\mathrm{KVanch}}\_{\text{unchanged}}\oplus\mathrm{KVupd}\_{n+1}.\end{aligned}$ | PC and FIFO update $\mathrm{KVupd}\_n$; $\mathrm{KVanch}$ is kept. FIFO evicts the oldest rows of $\mathrm{KVupd}\_n$, PC chooses the rows with the weakest link to the new chunk. Temporal RoPE is applied in both cases. | $\displaystyle \begin{aligned}&\underbrace{N\_{\rm pairs}=16}\_{\text{pairs}},\quad480\times832,\quad\mathrm{BF16}\\\\&\underbrace{d\_j=Q\_j^{\rm PC}-Q\_j^{\rm FIFO}}\_{Q:\ \text{MUSIQ at 15…19 s}}\\\\&\underbrace{5:11}\_{\text{wins PC : FIFO}},\quad\overline d=-1.384\\\\&H\_0:\Pr(d\_j>0\mid d\_j\ne0)=1/2\\\\&\underbrace{p\_{\rm value}=0.210>0.05}\_{\text{two-sided sign test}}\\\\&H\_0\text{ is not rejected}.\\\\&\text{By frames: the result depends on the scene.}\end{aligned}$<br><br>480p: videos and results<br>Frames: island · turtle<br>480p config · Previous 256p series |



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
