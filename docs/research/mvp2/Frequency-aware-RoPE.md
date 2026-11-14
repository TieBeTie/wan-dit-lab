# Frequency-aware RoPE: setup and implementation limits

Implemented as a constant spatial profile when the RoPE tables are prepared. It is not an additional layer, not switching RoPE off and not a new trainable module. The existing attention kernels receive the same cos/sin tables of the same shape.

Basis: [Lumina-Next, §2.2, Frequency-Aware Scaled RoPE](https://arxiv.org/html/2406.18583#S2.SS2). Below we use our explicit numbering of the real Wan channels, so that the reference frequency is determined by the physical wavelength condition.

For one spatial axis:

$$
\theta_j=10000^{-2j/D},\qquad
j_* = \frac{D}{2}\log_{10000}\frac{L}{2\pi},\qquad
\frac{2\pi}{\theta_{j_*}}=L.
$$

Here D=42, j=0,...,20, L is the trained length of the axis **in tokens**, that is 22 for H and 40 for W. The index j_* is real; there is no need to round it.

Constant profile:

$$
\theta'_j=\theta_j M^{\min(j/j_*,1)}.
$$

For M<1 this is equivalent to changing the base with a lower bound given by the Position Interpolation result. For M>1 the opposite direction of the bound is used, an upper bound given by the PI result. **This transfer to the stretching of a reduced grid is our adaptation, not a regime verified in the paper.** The original max cannot be kept literally for M>1: then the change of the low frequencies would exceed the chosen M.

If L≤2π, no pair makes a full turn on the trained axis; the profile uses uniform scaling. M=1 gives the original frequencies. The time table of the video stays the same.

For NTK the reference pair is the lowest one (j=20); here the reference is determined by the length of the trained axis, so the change saturates earlier. This does not guarantee an advantage over NTK and does not guarantee a geometric zoom.

## How to enable

In the object `export.rope` with `interpolate: true`:

```json
{
  "interpolate": true,
  "native_size": "704x1280",
  "scene_offset_frames": 8,
  "stretch": 0.2992905106483072,
  "by_parts": null,
  "ntk_aware": false,
  "frequency_aware": true
}
```

`stretch` is the exponent t, not the frequency multiplier: M_H=(22/16)^t, M_W=(40/29)^t for 512×928. The example above sets M_H=1.1. Frequency-aware cannot be enabled together with NTK or by-parts. Old configs do not have to add the new field; when it is absent the previous profiles are kept.

The profile is chosen once in Python when the tables for the export are created. No additional condition appears in the attention hot path; RoPE continues to be applied by the same kernel. Changing the profile changes the identity of the export and requires separate packages with the right constants.

Time-aware and Context Drop are not enabled by this implementation. Frequency-aware is constant across the denoising steps, so by itself it does not introduce a change of the phase convention between the saved keys of the video cache and the current queries.
