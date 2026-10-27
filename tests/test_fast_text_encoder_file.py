"""FastTextEncoder with its INT8 blocks in a mapped file gives the same embeddings as with them in pinned RAM."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.slow
def test_mapped_file_gives_the_pinned_embeddings(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from wan_dit_lab.models.text_encoder import FastTextEncoder
    prompts = ["A playful raccoon plays an electric guitar on a small stool in a cozy room."]
    dev = torch.device("cuda")
    pinned = FastTextEncoder(dev)
    want = pinned(prompts)
    del pinned
    path = tmp_path / "t5_int8.bin"
    made = FastTextEncoder(dev, weights_file=path)  # writes the file
    assert path.exists() and torch.equal(made(prompts), want)
    del made
    mapped = FastTextEncoder(dev, weights_file=path)  # reads it
    assert mapped.buffer is None  # no block kept in RAM between prompts
    assert torch.equal(mapped(prompts), want)
    from wan_dit_lab.offloading import offload
    if offload.RowFile.available():  # the same once more through the mapped file instead of the reads past the cache
        mapped.cache.clear()
        available = offload.RowFile.__dict__["available"]
        try:
            offload.RowFile.available = staticmethod(lambda: False)
            assert torch.equal(mapped(prompts), want)
        finally:
            offload.RowFile.available = available
