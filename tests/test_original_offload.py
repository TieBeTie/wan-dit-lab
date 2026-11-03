"""Memory transfers must preserve the reference block's values and deferred writes."""
import copy
import pytest
import torch
from scripts.experiments.original_offload import install_block_cache_offload, sample_latents, encode_serial


def test_taehv_output_is_streamed_as_rgb_frames():
    from scripts.experiments.original_offload import decode_taehv_frames
    calls = []
    def decoder(x):
        calls.append(tuple(x.shape))
        return torch.zeros((3, 1, 2, 2))
    frames = list(decode_taehv_frames(torch.zeros((1, 2, 48, 2, 2)), decoder))
    assert len(frames) == 2 and calls == [(48, 1, 2, 2)] * 2
    assert frames[0].shape == (1, 2, 2, 3) and frames[0].dtype == torch.uint8


def test_same_prompt_changes_do_not_become_scene_cuts():
    from scripts.experiments.original_offload import apply_prompt_cuts
    prefix = 'The scene transitions. '
    prompts = ['a', 'a', prefix + 'b', 'b', prefix + 'c']
    assert apply_prompt_cuts(prompts, [2, 2, 1], ['new', 'same', 'scene'], prefix) == ['a', 'a', 'b', 'b', prefix + 'c']


def test_serial_text_encoding_keeps_prompt_order_and_reuses_duplicates():
    calls = []
    def encoder(*, text_prompts):
        calls.append(text_prompts)
        return {'prompt_embeds': torch.tensor([len(x) for x in text_prompts]).reshape(-1, 1, 1)}
    actual = encode_serial(encoder, ['abc', 'd', 'abc'])
    assert actual['prompt_embeds'].flatten().tolist() == [3, 1, 3]
    assert calls == [['abc'], ['d']]


def test_reference_receives_one_video_with_whole_prompt_strings():
    class Pipeline:
        def inference(self, *, noise, text_prompts, return_latents):
            assert text_prompts == [['A cup pours water.', 'The cup stands upright.']]
            assert return_latents
            return noise
    value = object()
    assert sample_latents(Pipeline(), value, ['A cup pours water.', 'The cup stands upright.']) is value


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4, dtype=torch.bfloat16))

    def forward(self, x, kv_cache=None, crossattn_cache=None):
        kv_cache['k'].add_(1)
        crossattn_cache['is_init'] = True
        return x @ self.weight, (4, 4, {'new_k': x.clone(), 'action': 'direct_insert'})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer test')
def test_block_offload_keeps_bf16_values_and_returns_deferred_writes_to_cpu():
    block = Block().cuda()
    reference = copy.deepcopy(block)
    install_block_cache_offload(block, 'cuda')
    x = torch.arange(8, device='cuda', dtype=torch.bfloat16).reshape(2, 4)
    cache = {'k': torch.zeros((2, 4), dtype=torch.bfloat16)}
    backing = cache['k'].data_ptr()
    cross = {'is_init': False}
    ref_cache = {'k': cache['k'].cuda()}
    with torch.no_grad():
        expected, deferred = reference(x, kv_cache=ref_cache, crossattn_cache={})
        actual, update = block(x, kv_cache=cache, crossattn_cache=cross)
    assert torch.equal(actual, expected)
    assert cache['k'].device.type == 'cpu'
    assert cache['k'].data_ptr() == backing, 'offloading must keep the original CPU/file-backed storage'
    assert cache['k'].dtype == torch.bfloat16
    assert torch.equal(cache['k'], ref_cache['k'].cpu())
    assert cross['is_init']
    assert update[2]['new_k'].device.type == 'cpu'
    assert torch.equal(update[2]['new_k'], deferred[2]['new_k'].cpu())
