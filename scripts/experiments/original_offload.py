"""Official LongLive BF16 computation with CPU weight/KV placement for a small GPU.

This is an explicit reference experiment, not the production video runner.
The official sampler, transformer, BF16 weights, RoPE and Wan VAE are reused.
Placement and sequential VAE output are recorded as deviations from inference.py.
"""
import argparse
import atexit
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'reference/LongLive/LongLive2.0'
TORCH_LOAD = torch.load


def sample_latents(pipe, noise, prompts):
    return pipe.inference(noise=noise, text_prompts=[prompts], return_latents=True)


def encode_serial(encoder, prompts):
    cache = {}
    for prompt in prompts:
        if prompt not in cache:
            cache[prompt] = encoder(text_prompts=[prompt])['prompt_embeds']
    return {'prompt_embeds': torch.cat([cache[p] for p in prompts], dim=0)}


def decode_taehv_frames(latents, decoder):
    for i in range(latents.shape[1]):
        frames = decoder(latents[0, i:i+1].permute(1, 0, 2, 3))
        yield ((frames.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)


def apply_prompt_cuts(prompts, durations, cuts, prefix):
    if len(durations) != len(cuts) or sum(durations) != len(prompts):
        raise ValueError('Prompt schedule does not match cuts and generated chunks')
    result = list(prompts)
    at = 0
    for length, cut in zip(durations, cuts):
        if at and cut == 'same' and result[at].startswith(prefix):
            result[at] = result[at][len(prefix):]
        at += length
    return result


def shared_checkpoint_load(path, **unused):
    """Map immutable inference weights without Windows copy-on-write commit.

    This is only for trusted local torch.save zip checkpoints. Model code must
    not mutate these shared weights; dtype conversion creates separate storage.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    path = Path(path)
    with FakeTensorMode():
        metadata = TORCH_LOAD(path, map_location='cpu', weights_only=False)
    data = torch.from_file(str(path), shared=True, size=path.stat().st_size, dtype=torch.uint8)
    def materialize(value):
        if torch.is_tensor(value):
            offset = value.untyped_storage()._checkpoint_offset
            offset = offset // value.element_size() + value.storage_offset()
            return torch.empty(0, dtype=value.dtype).set_(data.untyped_storage(), offset, value.shape, value.stride())
        if isinstance(value, dict):
            return {k: materialize(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(materialize(v) for v in value)
        if isinstance(value, list):
            return [materialize(v) for v in value]
        return value
    return materialize(metadata)


def transfer_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)  # deliberately no dtype conversion
    if isinstance(value, dict):
        return {k: transfer_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(transfer_tree(v, device) for v in value)
    if isinstance(value, list):
        return [transfer_tree(v, device) for v in value]
    return value


def install_block_cache_offload(block, device):
    original = block.forward

    def forward(*args, **kwargs):
        caches = []
        for name in ('kv_cache', 'crossattn_cache'):
            cache = kwargs.get(name)
            if cache is not None:
                caches.append((cache, dict(cache)))
                cache.update(transfer_tree(cache, device))
        try:
            result = original(*args, **kwargs)
            # The model commits these writes after all blocks. Do not retain
            # 30 layers of GPU update payloads while their caches live on CPU.
            if isinstance(result, tuple) and len(result) == 2:
                result = result[0], transfer_tree(result[1], 'cpu')
            return result
        finally:
            for cache, backing in caches:
                for key, value in list(cache.items()):
                    original_value = backing.get(key)
                    if torch.is_tensor(value) and torch.is_tensor(original_value):
                        original_value.copy_(value)
                        cache[key] = original_value
                    else:
                        cache[key] = transfer_tree(value, 'cpu')

    block.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--chains', nargs='+', default=['teacup_action_hold', 'astronaut_apple_action_hold'])
    parser.add_argument('--placement-probe', action='store_true', help='One chunk using local dense S2 weights; not a quality reference')
    parser.add_argument('--profile-transfers', action='store_true', help='Measure one sample and stop before decode')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if args.placement_probe:
        config['checkpoints']['generator_ckpt'] = str(ROOT / 'checkpoints/LongLive-2.0-5B-S2/model_te.pt')
        config['num_output_frames'] = 8
        config['output_folder'] = str(Path(config['output_folder']).parent / 'placement-probe')
        args.chains = ['teacup_action_hold']
    checkpoint = Path(config['checkpoints']['generator_ckpt'])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if config.get('model_quant') or config.get('fp8_quant') or config['inference'].get('kv_quant'):
        raise ValueError('This reference experiment must remain unquantized BF16.')
    # Reference modules default to optional fused kernels in this revision.
    for name in ('LLV2_TRITON_ROPE', 'LLV2_TRITON_ADALN', 'LLV2_FREQS_I_CACHE', 'LLV2_USE_TE_ATTN', 'LLV2_USE_FA4'):
        os.environ[name] = '0'
    output = Path(config['output_folder'])
    output.mkdir(parents=True, exist_ok=True)
    import psutil
    memory = {'peak_rss_bytes': 0, 'peak_commit_bytes': 0, 'peak_cuda_allocated_bytes': 0, 'peak_cuda_reserved_bytes': 0}
    stopped = threading.Event()
    def sample_memory():
        while not stopped.is_set():
            info = psutil.Process().memory_info()
            memory['peak_rss_bytes'] = max(memory['peak_rss_bytes'], getattr(info, 'peak_wset', info.rss))
            memory['peak_commit_bytes'] = max(memory['peak_commit_bytes'], getattr(info, 'peak_pagefile', info.vms))
            if torch.cuda.is_initialized():
                memory['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                memory['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            (output / 'memory.json').write_text(json.dumps(memory, indent=2), encoding='utf-8')
            stopped.wait(1)
    threading.Thread(target=sample_memory, daemon=True).start()
    atexit.register(stopped.set)
    os.chdir(output.parent / 'original-workspace')
    sys.path.insert(0, str(ROOT))
    from scripts.conversion.prepare_live import compiler_environment
    os.environ.update(compiler_environment())
    sys.path.insert(0, str(REFERENCE))
    from accelerate import init_empty_weights
    from omegaconf import OmegaConf
    from utils.config import normalize_config
    from utils.memory import DynamicSwapInstaller
    from utils import wan_5b_wrapper as wrapper
    from pipeline import CausalDiffusionInferencePipeline
    from wan_5b.modules.causal_model import CausalWanModel
    from wan_5b.modules import causal_model, attention as reference_attention
    from utils.dataset import MultiTextConcatDataset
    if not (reference_attention.FLASH_ATTN_2_AVAILABLE or reference_attention.FLASH_ATTN_3_AVAILABLE):
        # Cross-attention directly calls FA2 in this revision; use the
        # reference's own dense SDPA fallback when FA2 has no local build.
        causal_model.flash_attention = reference_attention.attention

    torch.set_grad_enabled(False)
    torch.manual_seed(config['logging']['seed'])
    snapshot = {'config': config, 'placement_probe_not_quality_reference': args.placement_probe,
                'reference_revision': subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip(),
                'placement': 'CPU mmap weights; BF16 KV on CPU; one block cache on CUDA; no numerical quantization',
                'decode': 'TAEHV Wan 2.2, one latent frame at a time, CPU MP4 output',
                'attention': 'official default backend available in this environment; no Sage override',
                'torch': torch.__version__, 'status': 'loading', 'videos': []}
    def save():
        (output / ('run-' + '-'.join(args.chains) + '.json')).write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    save()
    state = shared_checkpoint_load(checkpoint)
    weights = state['generator']
    weights = {k.removeprefix('model.'): v for k, v in weights.items()}
    model_config = json.loads((ROOT / 'checkpoints/Wan2.2-TI2V-5B/config.json').read_text())

    def build_generator(cls, *unused, **overrides):
        accepted = inspect.signature(cls.__init__).parameters
        options = {k: v for k, v in model_config.items() if k in accepted}
        options.update(overrides)
        with init_empty_weights():
            model = cls(**options)
        model.load_state_dict(weights, strict=True, assign=True)
        return model

    make_t5 = wrapper.umt5_xxl
    def build_t5(*a, **kw):
        kw['device'] = 'meta'  # _t5 ends with .to(device); checkpoint assigns CPU storage below
        with init_empty_weights():
            return make_t5(*a, **kw)

    load_state = torch.nn.Module.load_state_dict
    def assign(module, state_dict, *a, **kw):
        kw['assign'] = True
        return load_state(module, state_dict, *a, **kw)

    def mapped_load(*a, **kw):
        return shared_checkpoint_load(a[0])

    c = normalize_config(OmegaConf.create(config))
    with ExitStack() as stack:
        stack.enter_context(patch.object(CausalWanModel, 'from_pretrained', classmethod(build_generator)))
        stack.enter_context(patch.object(wrapper, 'umt5_xxl', build_t5))
        stack.enter_context(patch.object(torch.nn.Module, 'load_state_dict', assign))
        stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self, *a, **kw: self))
        stack.enter_context(patch.object(torch, 'load', mapped_load))
        pipe = CausalDiffusionInferencePipeline(c, device=torch.device('cuda'), vae=torch.nn.Identity())
    pipe.to(dtype=torch.bfloat16)
    for component in (pipe.generator, pipe.text_encoder, pipe.vae):
        DynamicSwapInstaller.install_model(component, device=torch.device('cuda'))
    text_forward = pipe.text_encoder.forward
    pipe.text_encoder.forward = lambda text_prompts: encode_serial(text_forward, text_prompts)
    for block in pipe.generator.model.blocks:
        install_block_cache_offload(block, 'cuda')

    def allocate_kv(batch_size, dtype, device):
        # Same dense cache fields as the official initializer. CFG=1 never
        # reads the negative branch, so do not allocate its duplicate cache.
        if pipe.guidance_scale != 1:
            raise ValueError('CPU reference currently supports official guidance_scale=1 only')
        size = pipe.local_attn_size * pipe.frame_seq_length
        chunk = pipe.num_frame_per_block * pipe.frame_seq_length
        # Shared file backing avoids reserving another ~10 GiB of Windows
        # private commit. The values remain BF16 and are mapped through RAM.
        cache_path = output / '.kv-cache.bin'
        with cache_path.open('wb'):
            pass
        shape = (30, 2, batch_size, size, 24, 128)
        count = 1
        for n in shape:
            count *= n
        backing = torch.from_file(str(cache_path), shared=True, size=count, dtype=dtype).reshape(shape)
        pipe.kv_cache_pos = [dict(k=backing[i, 0],
            v=backing[i, 1], quantized=False,
            block_token_size=chunk, max_blocks=size // chunk, num_heads=24, num_filled_blocks=0,
            global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]),
            pinned_start=torch.tensor([-1]), pinned_len=torch.tensor([0])) for i in range(30)]
        pipe.kv_cache_neg = None
    pipe._initialize_kv_cache = allocate_kv
    cross_init = pipe._initialize_crossattn_cache
    pipe._initialize_crossattn_cache = lambda batch_size, dtype, device: cross_init(batch_size, dtype, 'cpu')
    dataset = MultiTextConcatDataset(c.data_path, num_blocks=c.num_output_frames // 8, deterministic=True)
    folder_ids = [x.name for x in dataset._folders]
    for id in args.chains:
        prompts = dataset[folder_ids.index(id)]['prompts']
        schedule = config.get('execution', {}).get('prompt_schedule', {})
        if id in schedule:
            spec = schedule[id]
            prompts = apply_prompt_cuts(prompts, spec['chunks'], spec['cuts'], dataset.scene_cut_prefix)
        torch.manual_seed(config['logging']['seed'])
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        noise = torch.randn((1, c.num_output_frames, 48, 44, 80), device='cuda', dtype=torch.bfloat16)
        print('BEGIN', id, flush=True)
        if args.profile_transfers:
            from scripts.experiments.transfer_probe import TransferProbe
            with TransferProbe() as transfers:
                latents = sample_latents(pipe, noise, prompts)
            report = transfers.report()
            (output / 'transfer-profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            snapshot['status'] = 'transfer_profile_complete'
            save()
            print(json.dumps(report), flush=True)
            return
        latents = sample_latents(pipe, noise, prompts)
        torch.save(latents.cpu(), output / f'{id}-latents.pt')
        pipe.clear_cache()
        torch.cuda.empty_cache()
        # TAEHV uses the same Wan 2.2 latent layout and streams RGB frames.
        path = output / f'{id}.mp4'
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '1280x704', '-r', '24', '-i', '-', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)
        from wan_dit_lab.models.vae import TAEHVStream
        decoder = TAEHVStream('cuda', checkpoint=ROOT / 'reference/taehv/taew2_2.pth')
        for pixels in decode_taehv_frames(latents, decoder):
            process.stdin.write(pixels.contiguous().cpu().numpy().tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')
        decoder.close()
        snapshot['videos'].append({'id': id, 'seconds': time.time()-started,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'prompts_per_chunk': prompts})
        save()
        del latents, noise
        print('DONE', path, flush=True)
    snapshot['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
