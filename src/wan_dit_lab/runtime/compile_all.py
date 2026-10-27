"""torch.compile (inductor) for what runs around the compiled blocks of the DiT: the model's own input and output
layers, and the blocks of the TAEHV decoder. Each module's forward is compiled in place, so the modules keep their
types (TAEHVStream tells its memory blocks by their class)."""

import torch


def compile_model_edges(model, mode="default", options=None):
    """The patch, text and time embeddings and the head of a WanModel."""
    settings = dict(options=options) if options else dict(mode=mode)
    for m in (model.patch_embedding, model.text_embedding, model.time_embedding, model.time_projection, model.head):
        m.forward = torch.compile(m.forward, **settings)


def compile_decoder(stream, mode="default"):
    """Every block of a TAEHVStream's decoder. Measured at 512x928: slower than eager (0.123 s a chunk against
    0.106): the blocks of one class share one compiled function, recompiled for each instance up to the limit of 8,
    the rest falls back to eager with the overhead of the guards. Not used."""
    for block in stream.decoder:
        block.forward = torch.compile(block.forward, mode=mode)
