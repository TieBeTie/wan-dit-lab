import hashlib
import torch
from scripts.experiments.original_offload import shared_checkpoint_load


def test_file_mapping_preserves_values_views_and_file_when_module_casts(tmp_path):
    path = tmp_path / 'weights.pt'
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    state = {'generator': {'weight': base, 'slice': base[:, 1:4],
                           'bf16': base.to(torch.bfloat16)}, 'format': 'test'}
    torch.save(state, path)
    before = hashlib.sha256(path.read_bytes()).digest()
    loaded = shared_checkpoint_load(path)
    assert loaded['format'] == 'test'
    for name, value in state['generator'].items():
        assert torch.equal(value, loaded['generator'][name])
        assert value.dtype == loaded['generator'][name].dtype
    assert loaded['generator']['weight'].untyped_storage().data_ptr() == loaded['generator']['slice'].untyped_storage().data_ptr()
    module = torch.nn.Linear(6, 4, bias=False)
    module.load_state_dict({'weight': loaded['generator']['weight']}, assign=True)
    module.to(dtype=torch.bfloat16)
    assert hashlib.sha256(path.read_bytes()).digest() == before
