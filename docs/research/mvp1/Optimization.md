<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Optimization: speed and memory

How can we bring generation close to playback speed while keeping an acceptable image?

$$
\underbrace{\rho=\frac{T_{\rm generation}}{T_{\rm video}}}_{\text{seconds of work per second of video}},\qquad
\underbrace{\rho\le1}_{\text{continuous playback without buffer exhaustion}},\qquad
\underbrace{M_{\rm peak}\le M_{\rm available}}_{\text{fits in the GPU}}
$$

| Transformation | Context → computation | Idea | [Observations](file:///C:/Inner%20Firmament/Growth%20Space/Projects/wan-dit-lab/outputs/experiments/2026-10-17/01_%D0%A0%D0%B0%D1%81%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5_256p_BF16_60%D1%81%D0%B5%D0%BA) |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle (H,W,\mathrm{shift})\mapsto(H^{\prime},W^{\prime},\mathrm{shift}^{\prime})$<br>[Scaling Rectified Flow, §5.3.2, (23)](https://arxiv.org/html/2403.03206#S5.SS3.SSS2) | $\displaystyle \begin{aligned}&\text{We reduce the frame; the noise schedule is changed separately.}\\&\underbrace{\mathrm{scaleArea}=\sqrt{\frac{H^{\prime}W^{\prime}}{HW}}}_{\text{scale by area}},\quad\underbrace{\mathrm{shift}^{\prime}=\mathrm{scaleArea}\,\mathrm{shift}}_{\text{new shift}}\\&\sigma_{\rm shift}(u)=\frac{\mathrm{shift}\,u}{1+(\mathrm{shift}-1)u},\quad\underbrace{u\in[0,1]}_{\text{schedule coordinate}}\\&(480,832,5)\mapsto(256,448,2.68)\\&\text{This is a schedule choice to be tested, not fine-tuning of the model.}\end{aligned}$ | Fewer spatial tokens; we check quality on videos. | BF16, 256×448, 60 s: ρ ≈ 1.824 at shift 2.68; peak allocated 4.77 GiB. The user chose this variant. |
| <span class="wan-kept">Applied</span><br>$\displaystyle \operatorname{TextKV}(\mathrm{ctx})\mapsto\mathrm{KVtext}$<br>[Basis: Attention Is All You Need, §3.2](https://arxiv.org/abs/1706.03762)<br>Caching is implemented by us. | $\displaystyle \begin{aligned}&\text{With an unchanged prompt, the text K/V are the same across steps.}\\&\underbrace{\mathrm{KVtext}=\{K_\ell^{\rm text}(\mathrm{ctx}),V_\ell^{\rm text}(\mathrm{ctx})\}_\ell}_{\text{computed once for a given text}}\\&A_\ell=\operatorname{Attn}(Q_\ell,K_\ell^{\rm text},V_\ell^{\rm text})\\&\text{When the text changes, we update the text cache.}\end{aligned}$ | We remove the repeated computation of the text K/V. | Enabled in the compared runs. There is no separate measurement of the gain here. |

Precision of the weights and of the KV storage — [[docs/research/mvp1/Quantization|Quantization · MVP1]].


The numbers refer to the steady state of the corresponding runs; loading and compilation are excluded. Peak allocated denotes the PyTorch metric.
