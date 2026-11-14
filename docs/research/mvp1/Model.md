<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Model

## General notation

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

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


## Text and decoder

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

Consider the input and output of the DiT: the prompt is turned into vectors, the latent chunk into frames.

$$
\underbrace{\mathrm{prompt}_j}_{\text{prompt}}\xrightarrow{\mathrm{T5}}\underbrace{\mathrm{ctx}_j}_{\text{prompt vectors}},\qquad
\underbrace{\mathrm{lat}_{n+1}}_{\text{latent chunk}}\xrightarrow{\mathrm{dec}}\underbrace{\mathrm{frame}_{n+1}}_{\text{chunk frames}},\qquad
\underbrace{\tfrac{H}{16}\times\tfrac{W}{16}}_{\text{latent grid of an }H\times W\text{ frame}}.
$$

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\mathrm{T5}=\mathrm{UMT5\mbox{-}XXL}\\[5pt]\mathrm{prompt}_j\mapsto\mathrm{ctx}_j\end{gathered}$<br>[Wan, §4.1–4.2](https://arxiv.org/html/2503.20314v1) | $\displaystyle \begin{aligned}&\text{Consider the text condition.}\\[6pt]&\underbrace{\mathrm{prompt}_j\xrightarrow{\mathrm{T5}}\mathrm{ctx}_j}_{\text{once per prompt}}\\[8pt]&\underbrace{\mathrm{KVtext}_j=\mathrm{KV}^{\rm cross}(\mathrm{ctx}_j)}_{\text{read by the cross-attention of every DiT block}}\\[7pt]&\text{T5 is not part of the chunk loop.}\end{aligned}$ | The cost of T5 is the prompt-change latency, not FPS.<br>The encoder gets VRAM last, after the DiT weights and KV. | $\displaystyle \begin{aligned}&\underbrace{W_{\rm T5}:\mathrm{BF16}\mapsto\mathrm{INT8}}_{\text{E30: }704\times1280,\ 8\text{ chunks}}\\[6pt]&T_{\rm prompt}:2.86\to1.63\ \mathrm{s}\\[5pt]&\mathrm{FPS}:5.41\ \text{and}\ 5.58\\[5pt]&\text{INT8: on viewing, the prompt is matched worse.}\\[5pt]&\text{Not measured with a metric.}\end{aligned}$<br>MVP1: INT8. MVP2: BF16.<br>Measurement, E30 |
| <span class="wan-kept">Applied</span><br>$\displaystyle \begin{gathered}\mathrm{dec}:\ \mathrm{Wan\ VAE}\mapsto\mathrm{TAEHV}\\[5pt]\mathrm{lat}_{n+1}\mapsto\mathrm{frame}_{n+1}\end{gathered}$<br>[TAEHV](https://github.com/madebyollin/taehv) | $\displaystyle \begin{aligned}&\text{Consider the decoding of a finished chunk.}\\[6pt]&\underbrace{\mathrm{lat}_{n+1}\xrightarrow{\mathrm{TAEHV}}\mathrm{frame}_{n+1}}_{\text{once per chunk, after the DiT}}\\[8pt]&\underbrace{F_{\rm chunk}=8\mapsto32\ \text{frames}}_{\text{4 frames per latent frame}}\\[7pt]&\underbrace{\mathrm{state}_{\rm dec}}_{\text{streaming decoder state between chunks}}\end{aligned}$ | A small decoder instead of Wan VAE: the main cost of a chunk remains the DiT.<br>The latent and KV do not depend on the decoder, so it can be changed separately from the rest. | $\displaystyle \begin{aligned}&\underbrace{T_{\rm dec}\approx0.1\ \mathrm{s}\ \text{out of}\ T_{\rm chunk}=2.39\ \mathrm{s}}_{544\times960,\ \text{MVP2 working point}}\\[6pt]&\text{Decoder on its own stream: FPS does not grow.}\\[5pt]&\text{H08: both decoders show light-blue snow.}\\[5pt]&\text{Not measured: }\mathcal L\text{ of TAEHV vs Wan VAE.}\end{aligned}$<br>Decoder comparison, H08 · Version taew2_2, E13 · Suspicion at the peak, E29 · Own stream |

## Before choosing LongLive

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| $\displaystyle \begin{gathered}\Phi_{\rm Wan}:\\(\epsilon_{0:N-1},\mathrm{ctx})\\\mapsto\mathrm{lat}_{0:N-1}\end{gathered}$<br>[Wan, §4.1–4.2](https://arxiv.org/html/2503.20314v1) | $\displaystyle \begin{aligned}&\text{Consider the base generation of a whole clip.}\\[6pt]&\underbrace{\epsilon_{0:N-1}}_{\text{noise for all chunks}}\xrightarrow{\Phi_{\rm Wan}}\underbrace{\mathrm{lat}_{0:N-1}}_{\text{all chunks of the clip}}\\[8pt]&\mathrm{lat}_{0:N-1}\xrightarrow{\rm VAE}\mathrm{video}\\[6pt]&\text{Denoising processes the chunks of the clip jointly.}\end{aligned}$ | We start with generating a whole clip; continuing beyond it requires a separate mechanism. | [Architecture and VAE](https://arxiv.org/html/2503.20314v1) |
| $\displaystyle \begin{gathered}\mathrm{Wan}\mapsto\mathrm{Self\mbox{-}Forcing}\\[5pt]\Phi_{\rm SF}:(\mathrm{KVvid}_n,\mathrm{ctx},\epsilon_{n+1})\\\mapsto(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})\end{gathered}$<br>[Self Forcing, §3.1–3.4](https://arxiv.org/html/2506.08009v1) | $\displaystyle \begin{aligned}&\text{Consider training the next chunk on the previous ones.}\\[6pt]&\underbrace{\mathrm{lat}_{<n}^{\rm data}}_{\text{context from the training data}}\mapsto\underbrace{\widehat{\mathrm{lat}}_{<n}}_{\text{generated by the model itself}}\\[8pt]&p_{\rm SF}(\mathrm{lat}_{0:N-1}\mid\mathrm{ctx})\\[3pt]&\qquad=\prod_{n=0}^{N-1}p_{\rm SF}(\mathrm{lat}_n\mid\mathrm{lat}_{<n},\mathrm{ctx})\\[8pt]&\underbrace{\min_{W_{\rm SF}}\mathcal L_{\rm train}(p_{\rm SF},p_{\rm data})}_{\text{we bring the distributions of whole clips closer}}\\[5pt]&\underbrace{p_{\rm data}}_{\text{distribution of real clips}}\\[8pt]&\text{KV keeps the context of the already generated chunks.}\end{aligned}$ | The authors train the model on its own previous outputs, as in generation; this reduces the mismatch between training and use.<br>We plugged in the ready Self-Forcing DMD weights; the ring, FIFO/PC are our memory work. | $\displaystyle \begin{aligned}&\text{E10: the weights are the same, only the window was changed.}\\[5pt]&(F_{\rm anch},F_{\rm win}):(10,21)\to(3,12)\\[5pt]&\mathrm{FPS}:1.42\to1.86\\[5pt]&M_{\rm VRAM}^{\rm peak}:7.0\to4.6\ \mathrm{GiB}\end{aligned}$<br>Window, E10 · Our KV cache · Four ways to change the prompt, E04 |

## After choosing LongLive

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| $\displaystyle \begin{gathered}W_{\rm SF}\mapsto W_{\rm LL\,1.3B}\\[5pt]\Phi_{\rm LL}:(\mathrm{KVvid}_n,\mathrm{ctx},\epsilon_{n+1})\\\mapsto(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})\end{gathered}$<br>[LongLive, §3](https://arxiv.org/html/2509.22622v1) | $\displaystyle \begin{aligned}&\text{Consider long continuation under limited memory.}\\[6pt]&\underbrace{\mathrm{KVvid}_n=\mathrm{KVanch}\oplus\mathrm{KVupd}_n}_{\text{initial anchor and a short window}}\\[8pt]&(\mathrm{KVvid}_n,\mathrm{ctx},\epsilon_{n+1})\\[3pt]&\qquad\xrightarrow{\Phi_{\rm LL}}(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})\\[8pt]&\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(3,3,12)}_{\text{our variant of LongLive-1.3B}}\\[7pt]&\text{The window and the anchor are included in training on long continuations.}\end{aligned}$ | Training on long continuations makes it possible to use a short window and a kept initial anchor.<br>Changing the prompt via KV recomputation is covered in [[Scene-change]]. | $\displaystyle \begin{aligned}&\underbrace{480\times832,\quad\mathrm{FPS}=1.75\text{–}1.93}_{\text{E11: two prompt-change modes}}\\[8pt]&\underbrace{60\ \mathrm{s}}_{\text{E12: no drift noted on viewing}}\\[5pt]&\text{A second cat appeared on the third prompt.}\end{aligned}$<br>Comparison, E11 · User rating, H14 · Continuation, E12 |
| <span class="wan-kept">MVP1 result</span><br>$\displaystyle \begin{gathered}W_{\rm LL\,1.3B}\mapsto W_{\rm LL\,2.0\,5B\,S2}\end{gathered}$<br>[LongLive 2.0](https://arxiv.org/abs/2605.18739) | $\displaystyle \begin{aligned}&\text{Consider the ready LongLive 2.0 weights for two steps.}\\[6pt]&\underbrace{\mathrm{Wan\,2.2}\longrightarrow\mathrm{LongLive\,2.0}}_{\text{by the authors: direct AR fine-tuning}}\\[8pt]&\underbrace{N_{\rm steps}:4\mapsto2}_{\text{ready distilled variant S2}}\\[8pt]&\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)}_{\text{chosen for MVP1}},\quad\mathrm{shift}=5\\[7pt]&(\mathrm{KVvid}_n,\mathrm{ctx},\epsilon_{n+1})\\[3pt]&\qquad\xrightarrow{\Phi_{\rm continue}^{(1)}}(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})\end{aligned}$<br>[MVP1 config](../../../configs/live.json) | We moved to the ready 5B S2 model; the early experiments with Wan and Self-Forcing are kept above.<br>In E14 the cat's actions improved; transitions between scenes required porting the multi-shot sink and narrative RoPE in E15. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:4.6\to16.4}_{\text{E26: 2 steps, W4A4 and Sage 2 together}}\\[8pt]&480\times832,\quad M_{\rm GPU}^{\rm peak}=5.83\ \mathrm{GiB}\\[5pt]&\underbrace{6\ \text{chunks}}_{\text{on viewing, the scene is coherent}}\end{aligned}$<br>Model comparison, E14 · Scene change, E15 · Run, E26 |
