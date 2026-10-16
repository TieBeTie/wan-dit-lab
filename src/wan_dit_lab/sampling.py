"""Flow-matching sampling with classifier-free guidance.

The loop of the reference Wan text-to-video pipeline (wan/text2video.py, `generate`), with the Euler step of the
reference FlowMatchScheduler instead of the UniPC solver of upstream.
"""

import importlib.util

import torch

from wan_dit_lab.checkpoint import REFERENCE

_spec = importlib.util.spec_from_file_location("reference_scheduler", REFERENCE / "utils" / "scheduler.py")
_scheduler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scheduler)
FlowMatchScheduler = _scheduler.FlowMatchScheduler


def sample(model, noise, context, context_null, grid_sizes, freqs, steps, shift, guide_scale,
           dtype=torch.bfloat16, on_step=None):
    """Denoises `noise` [16, F, H, W] into latents of the same shape.

    The conditional and the unconditional branch of guidance go through the model as one batch of two, so the
    offloaded weights cross PCIe once per step, not twice.
    """
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(steps)
    contexts = torch.cat([context, context_null]).to(noise.device)
    sizes = torch.cat([grid_sizes, grid_sizes])
    x = noise
    for i, t in enumerate(scheduler.timesteps):
        t = t.to(noise.device)[None]
        with torch.no_grad(), torch.autocast(noise.device.type, dtype=dtype):
            v_cond, v_null = model(torch.stack([x, x]), t.expand(2), contexts, sizes, freqs).float()
        x = scheduler.step(v_null + guide_scale * (v_cond - v_null), t, x)
        if on_step is not None:
            on_step(i)
    return x
