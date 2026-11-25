<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Quantization · MVP2

What changes in the quantizer when we descend from the $704\times1280$ reference? Consider the quantizer of MVP1 as one factor of the configuration.

$$
\underbrace{\mathcal Q^{(1)}=(\mathrm{W4A4},\ \mathrm{INT8},\ \mathrm{FP8\ E4M3})}_{(W_{\rm DiT},\ W_{\rm T5},\ \mathrm{KVvid})\text{ — [[docs/research/mvp1/Quantization|MVP1]]}}
\qquad\longmapsto\qquad
\underbrace{\mathcal Q^{(2)}=\mathcal Q^{(1)}\ \text{except}\ W_{\rm T5}:\ \mathrm{INT8}\mapsto\mathrm{BF16}}_{\text{MVP2}}
$$

Now consider how far two clips of the same configuration and seed are from each other: below this distance the loss cannot rank quantizers.

$$
\underbrace{\mathcal L_{\rm floor}=\mathcal L\bigl(\mathrm{video}(\mathrm{cfg},s),\ \mathrm{video}'(\mathrm{cfg},s)\bigr)}_{\text{second run, or a }10^{-5}\text{ perturbation of the input}},\qquad
\underbrace{\mathcal L_{\rm ceil}=0.736}_{\text{another seed}}
$$

| Transformation | Context → computation | Idea | Observations |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Applied</span><br>$\displaystyle W_{\rm DiT}:\ \mathrm{W4A4}\xrightarrow{\rm id}\mathrm{W4A4}$<br>checked against FP8 at the reference | $\displaystyle \begin{aligned}&\text{Consider the checkpoint LongLive-2.0-5B-NVFP4-S2.}\\[5pt]&\underbrace{\theta=\arg\min\ \mathcal L_{\rm train}\bigl(\Phi(\cdot\,;\mathcal Q_{\rm NVFP4}(W))\bigr)}_{\text{trained and distilled with 4-bit weights and activations}}\\[5pt]&\text{So the reference path of this checkpoint is the 4-bit one, not FP32.}\end{aligned}$ | For this checkpoint closer to FP32 does not mean a better video. Not verified: NVFP4 with block scales in training versus our W4A4 SVDQuant. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:2.17\to5.41}_{\mathrm{FP8}\to\mathrm{W4A4},\ 704\times1280}\\[5pt]&\text{In viewing (grid 03.11): W4A4 is not worse than FP8.}\end{aligned}$<br>[E30](../mvp2/E30.%20Starting%20point%20W4A4%20704%C3%971280%20and%20postponed%20steps.md) |
| <span class="wan-kept">Applied</span><br>$\displaystyle W_{\rm T5}:\ \mathrm{INT8}\mapsto\mathrm{BF16}$ | $\displaystyle \begin{aligned}&\text{Consider the prompt vectors }\mathrm{ctx}=\mathrm{T5}(\mathrm{prompt};W_{\rm T5}).\\[5pt]&\text{Was: }\widehat W_{\rm T5}^{\rm INT8}.\quad\text{Now: }W_{\rm T5}^{\rm BF16}.\end{aligned}$ | We return the encoder to 16 bits and pay with the prompt delay. INT8 is postponed to the end of the descent. | $\displaystyle \begin{aligned}&\text{In viewing: with INT8 the video follows the prompt worse.}\\[5pt]&T_{\rm prompt}:1.63\to2.86\ \mathrm{s}\\[5pt]&\underbrace{\mathrm{FPS}:5.58\to5.41}_{704\times1280}\end{aligned}$<br>[E30](../mvp2/E30.%20Starting%20point%20W4A4%20704%C3%971280%20and%20postponed%20steps.md) |
| <span class="wan-kept">Applied</span> in experiments<br>$\displaystyle \textstyle\sum_{\rm parallel}\mapsto\sum_{\rm ordered}$<br>`kernels.lora_sum` | $\displaystyle \begin{aligned}&\text{Consider the output of the W4A4 layer as a sum of terms }a_{r}.\\[5pt]&\text{Was: }y=\textstyle\sum_r a_r\ \text{added by parallel threads, the order is not fixed.}\\[5pt]&\text{Now: the same terms are added in a fixed order.}\end{aligned}$ | Float addition is not associative, so the order changes the last bits and the clip diverges. A fixed order makes a repeat bit-identical; Live keeps the parallel sum (`atomic`). | $\displaystyle \begin{aligned}&\underbrace{\mathcal L_{\rm floor}:0.26\,/\,0.18\to0}_{\text{second run, fox / cup}}\\[5pt]&\underbrace{-2\%\ \mathrm{FPS}}_{\text{ordered}},\qquad\underbrace{+1.7\%\ \mathrm{FPS}}_{\text{atomic}}\end{aligned}$<br>[Report, §31](../../report-2026-11-11-loss-704-608-for-fable.md) |
| Measurement<br>$\displaystyle \mathcal L_{\rm floor}(\mathcal Q)$ | $\displaystyle \begin{aligned}&\text{Consider a }10^{-5}\text{ perturbation of the input at }704\times1280.\\[5pt]&\mathcal L_{\rm floor}=\mathcal L\bigl(\mathrm{video}(x),\mathrm{video}(x+10^{-5}\xi)\bigr)\end{aligned}$ | Divergence of close inputs is a property of the model; the 4-bit path doubles it. A paired loss below the floor is not a result. | $\displaystyle \begin{aligned}&\underbrace{0.12}_{\mathrm{FP8}+\mathrm{SDPA}}\to\underbrace{0.18}_{\mathrm{FP8}+\mathrm{Sage}}\to\underbrace{0.33}_{\mathrm{W4A4}+\mathrm{Sage}}\end{aligned}$<br>[Report, part 3](../../report-2026-11-11-loss-704-608-for-fable.md) |
| Not applied: the gap is not from calibration<br>$\displaystyle s_{704}\mapsto s_{608}$ | $\displaystyle \begin{aligned}&\text{Consider the smoothing }s\text{ of W4A4, calibrated at }704\times1280.\\[5pt]&\text{We compare the loss at }608\text{ with exact FP8 weights and with W4A4.}\end{aligned}$ | If W4A4 lost because of a foreign calibration, FP8 would be closer to the reference. It is not, so one calibration serves all resolutions. | $\displaystyle \begin{aligned}&\underbrace{\overline{\mathcal L}_1:0.382\ \text{vs}\ 0.373}_{\mathrm{FP8}\ \text{vs}\ \mathrm{W4A4},\ 608}\end{aligned}$<br>[Report, §31](../../report-2026-11-11-loss-704-608-for-fable.md) |

$$
\mathcal Q^{(2)}=\bigl(\underbrace{\mathrm{W4A4}}_{W_{\rm DiT}},\ \underbrace{\mathrm{BF16}}_{W_{\rm T5}},\ \underbrace{\mathrm{FP8\ E4M3}}_{\mathrm{KVvid}}\bigr)
$$

[Working point config](../../../configs/candidates/544x960-win32-live.json) · Experiment: [E30](../mvp2/E30.%20Starting%20point%20W4A4%20704%C3%971280%20and%20postponed%20steps.md)
