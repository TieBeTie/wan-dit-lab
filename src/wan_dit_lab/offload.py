import copy
import queue
import threading
from collections import deque
from itertools import cycle

import torch
from torch.nn.utils import vector_to_parameters


def module_on(row, like):
    """A copy of the module `like` whose parameters are views of `row`.

    The parameters need no gradient, and without one autocast does not cache their low-precision copies: the same
    parameter of a slot holds a new block after every copy, and a cached cast would keep the weights of the first.
    """
    module = copy.deepcopy(like).requires_grad_(False)
    vector_to_parameters(row, module.parameters())
    return module


class OnDevice:
    """All the blocks on the GPU, in `dtype`: the calls of BlockOffloader for a model that fits. get_next gives the
    blocks in order, release does nothing. Wan 1.3B takes 2.6 GB in bfloat16; nothing crosses PCIe while it runs."""

    def __init__(self, blocks, device, dtype=torch.bfloat16):
        self.blocks = blocks
        self.device = device
        self.dtype = dtype

    def start(self):
        for block in self.blocks:
            block.to(self.device, self.dtype).requires_grad_(False)
        self.order = cycle(self.blocks)

    def get_next(self):
        return next(self.order)

    def release(self, block):
        pass

    def stop(self):
        pass


class BlockOffloader:
    """Leaves the blocks where they are on the CPU (RAM or a memory-mapped checkpoint), keeps the next
    `ram_window` of them in pinned buffers and at most `window` on the GPU.

    Two stages: a thread copies the parameters of the next block into a free pinned buffer, the copy stream
    copies the buffer into a free GPU slot.
    """

    def __init__(self, blocks, window, device, ram_window=2):
        self.blocks = blocks
        self.window = window
        self.ram_window = ram_window
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.thread = None

    def start(self):
        self.stop()
        size = sum(p.numel() for p in self.blocks[0].parameters())
        dtype = next(self.blocks[0].parameters()).dtype
        self.pinned = [torch.empty(size, dtype=dtype, pin_memory=True) for _ in range(self.ram_window)]
        self.gpu = torch.empty((self.window, size), dtype=dtype, device=self.device)
        self.slots = [module_on(row, like=self.blocks[0]) for row in self.gpu]
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        self.free = queue.Queue()  
        self.filled = queue.Queue()  
        for buffer in self.pinned:
            self.free.put((buffer, None))
        self.thread = threading.Thread(target=self.fill, daemon=True)
        self.thread.start()
        self.loaded = deque()
        for slot in range(self.window):
            self.load(slot, after=None)

    def fill(self):
        """Stage 1, in its own thread: copies the parameters of the next block into a free pinned buffer."""
        for block in cycle(self.blocks):
            buffer, copied = self.free.get()
            if buffer is None:
                return
            if copied is not None:
                copied.synchronize()  # the copy to the GPU has finished reading this buffer
            torch.cat([p.detach().reshape(-1) for p in block.parameters()], out=buffer)
            self.filled.put(buffer)

    def load(self, slot, after):
        """Stage 2: enqueues the copy of the next block from its pinned buffer into `slot`, after the event `after`."""
        buffer = self.filled.get()
        with torch.cuda.stream(self.copy_stream):
            if after is not None:
                self.copy_stream.wait_event(after)
            self.gpu[slot].copy_(buffer, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
        self.free.put((buffer, copied))
        self.loaded.append((slot, copied))

    def get_next(self):
        slot, copied = self.loaded.popleft()
        torch.cuda.current_stream().wait_event(copied)
        return self.slots[slot]

    def release(self, block):
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        self.load(self.slots.index(block), after=done)

    def stop(self):
        if self.thread is not None:
            self.free.put((None, None))
            self.thread.join()
            self.thread = None
