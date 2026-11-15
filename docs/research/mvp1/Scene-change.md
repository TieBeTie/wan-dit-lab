<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Scene change: hybrid control

$$
\begin{array}{ll}
\begin{gathered}
\underbrace{\mathrm{prompt}_0}_{\text{old prompt}},\qquad\underbrace{\mathrm{prompt}_1}_{\text{new prompt}}\\
\mathrm{prompt}_j\xrightarrow{\rm text\ encoder}\underbrace{\mathrm{ctx}_j}_{\text{text vectors}}\\
\underbrace{\mathrm{KVvid}_j=\{K^{\rm self}_{\ell,j},V^{\rm self}_{\ell,j}\}_\ell}_{\text{video memory: self-attention KV}}\\
\underbrace{\mathrm{KVtext}_j=\{K^{\rm cross}_\ell(\mathrm{ctx}_j),V^{\rm cross}_\ell(\mathrm{ctx}_j)\}_\ell}_{\text{text cache: cross-attention KV}}\\
j\in\{0,1\},\qquad\underbrace{\ell=1,\ldots,30}_{\text{block index}}
\end{gathered}
&
\begin{gathered}
\text{Consider block }\ell\\
\begin{array}{ccl}
\underbrace{\mathrm{tok}_{\ell-1}}_{\text{video tokens}}&&\\
\downarrow&&\\
\boxed{\rm self\!\text{-}\!attention}&\xrightarrow{\rm write\ KV}&\mathrm{KVvid}_{\ell,j}\\
\downarrow&&\\
\boxed{\rm cross\!\text{-}\!attention\ (\mathrm{KVtext}_{\ell,j})}&&\\
\downarrow&&\\
\boxed{\rm FFN}&&\\
\downarrow&&\\
\mathrm{tok}_\ell&&
\end{array}
\end{gathered}
\end{array}
$$

$$
(\mathrm{KVvid}_0,\mathrm{KVtext}_0)\xrightarrow{\rm assistant}
\begin{cases}
(\varnothing,\mathrm{KVtext}_1),&\text{new scene},\\
\Phi(\mathrm{KVvid}_0,\mathrm{KVtext}_0),&\text{continuation}.
\end{cases}
$$

> Preliminary observations from a single clip. There is not yet enough comparative data; we will add observations on the right.

Φ denotes a transformation of the pair (video memory, text condition); its index specifies the way the prompt is changed.

| Transformation | Context → computation | Idea | Observations · one clip |
| :--- | :--- | :--- | :--- |
| $`\displaystyle \begin{gathered}\Phi_{\rm swap}\colon\\(\mathrm{KVvid}_0,\mathrm{KVtext}_0)\mapsto(\mathrm{KVvid}_0,\mathrm{KVtext}_1)\end{gathered}`$<br>[Comparison with the stored KV: LongLive, §4.4](https://arxiv.org/html/2509.22622v2#S4.SS4) | $`\displaystyle \begin{aligned}&\text{We replace the text condition and do not recompute the history:}\\&\mathrm{ctx}_0\to \mathrm{ctx}_1,\qquad\mathrm{KVtext}_0\leftarrow\mathrm{KVtext}_1,\qquad\mathrm{KVvid}_0\text{ is kept}\end{aligned}`$ | New text, the same video memory. | In this clip I see no noticeable difference from recache. |
| $`\displaystyle \begin{gathered}\Phi_{\rm recache}\colon\\(\mathrm{KVvid}_0,\mathrm{KVtext}_0)\mapsto(\mathrm{KVvid}_1,\mathrm{KVtext}_1)\end{gathered}`$<br>[KV-recache: LongLive](https://arxiv.org/html/2509.22622v2#S3.SS1) | $`\displaystyle \begin{aligned}&\text{Consider a repeated pass over the already generated video with a new prompt.}\\&\underbrace{\mathrm{lat}=(\mathrm{lat}_1,\ldots,\mathrm{lat}_m)}_{\text{stored latents of the generated video}},\qquad\underbrace{\mathrm{lat}_i}_{\text{internal representation of the video}}\xrightarrow{\rm decoder}\text{frames}\\&\text{The denoiser accepts latents with added noise:}\\&\underbrace{\mathrm{lat}_{\sigma}=(1-\sigma)\mathrm{lat}+\sigma\epsilon}_{\text{denoiser input}},\qquad\underbrace{\epsilon\sim\mathcal N(0,I)}_{\text{random noise}}\\&\text{Through the timestep we tell the model how much noise is in the input:}\\&\underbrace{\mathrm{step}}_{\text{timestep input of the model}}=1000\underbrace{\sigma}_{\text{noise coefficient}},\qquad \mathrm{step}=0\Rightarrow \mathrm{lat}_{\sigma}=\mathrm{lat}\\&\text{For recache we do not add noise: we take the finished latents at }\mathrm{step}=0.\\&\text{The repeated pass with the new text writes the self-attention K/V:}\\&\left(\underbrace{\mathrm{lat}}_{\text{the same finished latents}},\ \underbrace{\mathrm{step}=0}_{\text{no added noise}},\ \underbrace{\mathrm{ctx}_1}_{\text{new prompt vectors}}\right)\xrightarrow[W^{\rm DiT}=W^{\rm SelfForcing}]{\rm forward+write\ self\ KV}\underbrace{\mathrm{KVvid}_1}_{\text{new video memory}}\\\end{aligned}`$ | The same frames, new text: we rebuild the video memory from scratch. | In this clip I see no noticeable difference from swap. |
| $`\displaystyle \begin{gathered}\Phi_{\rm decay}\colon\\(\mathrm{KVvid}_0,\mathrm{KVtext}_0)\mapsto(\widetilde{\mathrm{KVvid}}_0(\mathrm{age}),\mathrm{KVtext}_1)\end{gathered}`$<br>[Echo-Forcing](https://arxiv.org/abs/2605.16003)<br>Our implementation of the decay idea. | $`\displaystyle \begin{aligned}&\text{Consider self-attention: how the old K/V influence the new chunk.}\\&\mathrm{Attn}(\underbrace{Q}_{\text{queries of the new chunk}},K,V)=\mathrm{softmax}(QK^{\mathsf T}/\sqrt{d_h})V,\quad\underbrace{d_h}_{\text{head size}}\\&\text{The first chunk after the prompt change gives a reference for comparison:}\\&\underbrace{K_0,V_0}_{\text{old memory}},\qquad\underbrace{K_{\rm cur},V_{\rm cur}}_{\text{K/V of the new chunk}},\qquad\underbrace{\mathrm{ref}}_{\text{last frame of the first new chunk}}\\&\text{We compare the old keys with the reference at the same places of the frame:}\\&\underbrace{\mathrm{patch}(i)}_{\text{the same patch of the reference frame}},\qquad\underbrace{P_{\rm sp}k}_{\text{spatial features of the key}}\\&\underbrace{\mathrm{diff}_i}_{\text{difference}}=1-\cos(P_{\rm sp}k_i^{\rm old},P_{\rm sp}k_{\mathrm{patch}(i)}^{\rm ref})\\&\text{A larger difference is turned into a higher forgetting rate:}\\&\underbrace{\mathrm{diff}_{\min},\mathrm{diff}_{\max}}_{\text{min/max over the old tokens of the block}},\qquad\underbrace{\epsilon_{\rm safe}>0}_{\text{protection against division by zero}}\\&\underbrace{\mathrm{decay}_i}_{\text{forgetting rate}}=\mathrm{decay}_{\min}+(\mathrm{decay}_{\max}-\mathrm{decay}_{\min})\frac{\mathrm{diff}_i-\mathrm{diff}_{\min}}{\mathrm{diff}_{\max}-\mathrm{diff}_{\min}+\epsilon_{\rm safe}},\quad\underbrace{\mathrm{decay}_{\min},\mathrm{decay}_{\max}}_{\text{given limits}}\\&\text{With each chunk we exponentially reduce the weight of the old row:}\\&\underbrace{w_i(\mathrm{age})=e^{-\mathrm{age}\,\mathrm{decay}_i}}_{\text{weight of the old row}},\quad\underbrace{D(\mathrm{age})=\mathrm{diag}_i w_i(\mathrm{age})}_{\text{weight matrix}},\quad\underbrace{\mathrm{age}}_{\text{chunks after the reference}}\\&\text{We apply the weights to working copies of K/V before attention:}\\&\underbrace{\widetilde{\mathrm{KVvid}}_0(\mathrm{age})=\{D(\mathrm{age})K_0,D(\mathrm{age})V_0\}}_{\text{temporary matrices for attention}},\qquad\underbrace{K_0,V_0}_{\text{stored unchanged}}\\&\text{We concatenate them with the K/V of the new chunk and compute attention:}\\&A_{\rm age}=\mathrm{Attn}\!\left(Q,\begin{bmatrix}D(\mathrm{age})K_0\\K_{\rm cur}\end{bmatrix},\begin{bmatrix}D(\mathrm{age})V_0\\V_{\rm cur}\end{bmatrix}\right)\end{aligned}`$ | The more an old token differs from the new scene, the faster we forget it. | The image «drifts». The cause is not yet established; an application error remains a hypothesis. |
| $`\displaystyle \begin{gathered}\Phi_{\rm blend}\colon\\(\mathrm{KVvid}_0,\mathrm{KVtext}_0)\mapsto(\mathrm{KVvid}_0,\mathrm{KVtext}_\alpha)\end{gathered}`$<br>[Basis: cross-attention](https://arxiv.org/abs/1706.03762)<br>Our K/V interpolation. | $`\displaystyle \begin{aligned}&\text{Consider cross-attention with a gradually changing text condition.}\\&\underbrace{Q^{\rm text}}_{\text{video queries}},\qquad\underbrace{(K_j^{\rm text},V_j^{\rm text})}_{\text{K/V of the text }\mathrm{ctx}_j}\\&\underbrace{N_{\rm blend}}_{\text{transition length in chunks}},\qquad r=1,\ldots,N_{\rm blend},\qquad\alpha_r=\min(r/N_{\rm blend},1)\\&\underbrace{\mathrm{KVtext}_\alpha=(1-\alpha)\mathrm{KVtext}_0+\alpha\mathrm{KVtext}_1}_{\text{interpolation of the text K/V}}=\{K_\alpha^{\rm text},V_\alpha^{\rm text}\}\\&\mathrm{CrossAttn}_\alpha=\mathrm{Attn}(Q^{\rm text},K_\alpha^{\rm text},V_\alpha^{\rm text}),\quad\mathrm{KVvid}_0\text{ is not recomputed}\end{aligned}`$ | We change the text condition smoothly while keeping the video memory. | The transition looks good visually. So far this is an impression from one clip. |
