import copy
import queue
import sys
import threading
import time
from collections import deque
from itertools import cycle

import torch


def module_on(row, like):
    """A copy of the module `like` whose parameters are views of `row`.

    The parameters need no gradient, and without one autocast does not cache their low-precision copies: the same
    parameter of a slot holds a new block after every copy, and a cached cast would keep the weights of the first.
    """
    # Copy structure, not the checkpoint's full CPU parameter payload for every slot.
    memo, offset = {}, 0
    for parameter in like.parameters():
        view = row[offset:offset + parameter.numel()].view_as(parameter)
        memo[id(parameter)] = torch.nn.Parameter(view, requires_grad=False)
        offset += parameter.numel()
    module = copy.deepcopy(like, memo).requires_grad_(False)
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
            # dtype None: the blocks keep their dtypes (already converted, e.g. FP8 buffers that a cast would undo)
            (block.to(self.device, self.dtype) if self.dtype is not None else block.to(self.device)).requires_grad_(False)
        self.order = cycle(self.blocks)

    def get_next(self):
        return next(self.order)

    def release(self, block):
        pass

    def stop(self):
        pass

    def close(self):
        """Drop this owner's references; the model may still own the same blocks."""
        self.order = None
        self.blocks = None


def block_tensors(block):
    """Every tensor a block computes with, in a fixed order: parameters, buffers, and the weight scales of its
    Fp8Linear layers, which are plain attributes (fp8.py)."""
    tensors = list(block.parameters()) + list(block.buffers())
    tensors += [m.scale for m in block.modules() if isinstance(getattr(m, "scale", None), torch.Tensor)]
    return tensors


class StreamedOnDevice:
    """OnDevice for a model a little larger than the GPU: the blocks stay on it as they are (FP8 in Live) except
    `streamed` of them, kept in pinned RAM and copied, on a stream of their own, into one of two GPU slots while the
    blocks before them compute. The streamed blocks are spread over the model, so that each copy has the time of
    several blocks to finish; the copy into a slot waits until the block that used it last has computed."""

    def __init__(self, blocks, device, streamed):
        self.blocks, self.device = blocks, device
        self.streamed = sorted(set(streamed))
        # two slots taken in turn: an even number, so that each block always comes back to the same slot
        assert len(self.streamed) >= 2 and len(self.streamed) % 2 == 0, "an even number of streamed blocks, 2 or more"

    def start(self):
        first = self.blocks[self.streamed[0]]
        self.slots = [copy.deepcopy(first).to(self.device).requires_grad_(False) for _ in range(2)]
        self.host = {}
        for i in self.streamed:  # a pinned copy of each, the GPU copy dropped
            self.host[i] = [t.detach().to("cpu").pin_memory() for t in block_tensors(self.blocks[i])]
            self.blocks[i].to("meta")
        self.copy_stream = torch.cuda.Stream(self.device)
        self.copied = [torch.cuda.Event() for _ in self.slots]
        self.freed = [torch.cuda.Event() for _ in self.slots]
        self.slot_of = {i: k % 2 for k, i in enumerate(self.streamed)}  # streamed blocks alternate between slots
        self.i = 0
        for k in (0, 1):
            self.copy(self.streamed[k], wait=False)

    def copy(self, i, wait=True):
        slot = self.slot_of[i]
        with torch.cuda.stream(self.copy_stream):
            if wait:
                self.copy_stream.wait_event(self.freed[slot])
            for target, source in zip(block_tensors(self.slots[slot]), self.host[i]):
                target.copy_(source, non_blocking=True)
            self.copied[slot].record(self.copy_stream)

    def modules(self):
        """The modules get_next can give: the blocks kept on the GPU and the two slots."""
        return [b for i, b in enumerate(self.blocks) if i not in self.slot_of] + self.slots

    def get_next(self):
        i, self.i = self.i, (self.i + 1) % len(self.blocks)
        if i not in self.slot_of:
            return self.blocks[i]
        slot = self.slot_of[i]
        torch.cuda.current_stream(self.device).wait_event(self.copied[slot])
        self.current = i
        return self.slots[slot]

    def release(self, block):
        if not any(block is slot for slot in self.slots):
            return
        i = self.current
        slot = self.slot_of[i]
        self.freed[slot].record(torch.cuda.current_stream(self.device))  # the slot is free once this block computed
        k = self.streamed.index(i)
        self.copy(self.streamed[(k + 2) % len(self.streamed)])  # the next streamed block of this slot

    def stop(self):
        pass

    def close(self):
        self.blocks = self.slots = self.host = None


class StreamedRows(StreamedOnDevice):
    """StreamedOnDevice for blocks that are views of one row of bytes each (a W4A4 or FP8 pack, the slot function
    of its format): `streamed` of them stay as rows in pinned RAM and go, row by row, into one of two GPU rows while
    the blocks before them compute; the others are on the GPU as given. Live keeps the VRAM it needs beside a
    browser and OBS this way (a W4A4 block is 92 MB: its copy, 8 ms, hides behind the 25 ms of the block before)."""

    def __init__(self, blocks, device, streamed, rows, slot):
        super().__init__(blocks, device, streamed)
        self.rows, self.make_slot = rows, slot

    def start(self):
        self.slot_rows = [torch.empty(self.rows.shape[1], dtype=torch.uint8, device=self.device) for _ in range(2)]
        self.slots = [self.make_slot(row) for row in self.slot_rows]
        self.host = {i: self.rows[i].clone().pin_memory() for i in self.streamed}
        self.copy_stream = torch.cuda.Stream(self.device)
        self.copied = [torch.cuda.Event() for _ in self.slots]
        self.freed = [torch.cuda.Event() for _ in self.slots]
        self.slot_of = {i: k % 2 for k, i in enumerate(self.streamed)}
        for i in self.streamed:  # the model's list names the slot a streamed block runs in
            self.blocks[i] = self.slots[self.slot_of[i]]
        self.i = 0
        for k in (0, 1):
            self.copy(self.streamed[k], wait=False)

    def copy(self, i, wait=True):
        slot = self.slot_of[i]
        with torch.cuda.stream(self.copy_stream):
            if wait:
                self.copy_stream.wait_event(self.freed[slot])
            self.slot_rows[slot].copy_(self.host[i], non_blocking=True)
            self.copied[slot].record(self.copy_stream)

    def modules(self):
        return [b for i, b in enumerate(self.blocks) if i not in self.slot_of] + self.slots


SECTOR = 4096  # what reads without the page cache align their offset, size and buffer to


class RowFile:
    """A file of rows of equal size, read row by row without the page cache (FILE_FLAG_NO_BUFFERING, Windows), so
    that the file takes no memory beyond the buffer it is read into. A memory-mapped file grew the working set of
    Live by 5.2 GB for each T5 prompt and pushed the tutor's pages out; read this way it grows by nothing, and faster
    (5.4 GB of T5 rows: 1.4 s against 3.7 s mapped), as sequential reads of whole rows keep the NVMe busy."""

    def __init__(self, path, row_bytes):
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN
        self.handle = k32.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x20000000 | 0x08000000, None)
        if self.handle in (None, wintypes.HANDLE(-1).value):
            raise OSError(ctypes.get_last_error(), f"CreateFileW {path}")
        self.k32, self.ctypes, self.wintypes = k32, ctypes, wintypes
        self.row_bytes = row_bytes
        # a row is read with the sectors around it: its buffer is that much longer
        self.span = -(-(row_bytes + 2 * SECTOR) // SECTOR) * SECTOR

    @staticmethod
    def available():
        return sys.platform == "win32"

    def buffer(self):
        """A buffer for one row: plain memory, page-aligned, registered with CUDA (pinned) until unpin(); unlike the
        pinned memory of torch, which keeps it for the next allocation, this goes back to Windows after a prompt."""
        raw = torch.empty(self.span + SECTOR, dtype=torch.uint8)
        start = -raw.data_ptr() % SECTOR
        aligned = raw[start:start + self.span]
        status = torch.cuda.cudart().cudaHostRegister(aligned.data_ptr(), aligned.numel(), 0)
        if int(status) != 0:
            raise RuntimeError(f"cudaHostRegister failed ({status})")
        return aligned

    @staticmethod
    def unpin(buffer):
        torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())

    def read(self, index, buffer):
        """Row `index` into `buffer` (from buffer()); returns the view of the row in it."""
        start = index * self.row_bytes
        low = start // SECTOR * SECTOR
        size = -(-(start + self.row_bytes - low) // SECTOR) * SECTOR
        self.k32.SetFilePointerEx(self.handle, low, None, 0)
        done, got = 0, self.wintypes.DWORD()
        while done < size:
            if not self.k32.ReadFile(self.handle, buffer.data_ptr() + done, min(size - done, 64 << 20),
                                     self.ctypes.byref(got), None):
                raise OSError(self.ctypes.get_last_error(), "ReadFile")
            if got.value == 0:  # the end of the file: its last sector was not whole
                break
            done += got.value
        return buffer[start - low:start - low + self.row_bytes]

    def close(self):
        if self.handle is not None:
            self.k32.CloseHandle(self.handle)
            self.handle = None


class BlockOffloader:
    """Leaves the blocks where they are on the CPU (RAM or a memory-mapped checkpoint), keeps the next
    `ram_window` of them in pinned buffers and at most `window` on the GPU.

    Two stages: a thread copies the parameters of the next block into a free pinned buffer, the copy stream
    copies the buffer into a free GPU slot.
    """

    def __init__(self, blocks, window, device, ram_window=2, trace=None, dtype=None,
                 resource_cache=None, resource_prefix="weights", packed=None, slot=None, rows=None, once=False):
        # slot: makes a block over one row of `packed` (fp8.fp8_slot for a block_fp8_v1 pack, rows of uint8);
        # without it the row is the parameters of the block in order (module_on)
        # rows: a RowFile the rows of `slot` are read from instead of `packed`, straight into the pinned buffers
        # once: one pass over the blocks (a T5 prompt), not the cycle of a denoising loop; nothing is read after it
        self.slot, self.rows, self.once = slot, rows, once
        if window < 1 or ram_window < 1:
            raise ValueError("Offload windows must be positive")
        self.blocks = blocks
        self.window = window
        self.ram_window = ram_window
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.thread = None
        self.trace = trace
        self.dtype, self.resource_cache, self.resource_prefix = dtype, resource_cache, resource_prefix
        self.source_names = []
        self.packed = packed
        if rows is not None and (slot is None or packed is not None or resource_cache is not None):
            raise ValueError("Rows from a file need a slot, and replace packed weights and the resource cache")
        expected = packed.shape[1] if slot is not None and packed is not None else \
            sum(p.numel() for p in blocks[0].parameters())
        if packed is not None and (tuple(packed.shape) != (len(blocks), expected)
                or packed.device.type != "cpu" or packed.dtype != (dtype or next(blocks[0].parameters()).dtype)):
            raise ValueError("Packed weights do not match the block slots")

    def start(self):
        self.stop()
        size = self.rows.row_bytes if self.rows is not None else self.packed.shape[1] if self.slot is not None \
            else sum(p.numel() for p in self.blocks[0].parameters())
        dtype = self.dtype or next(self.blocks[0].parameters()).dtype
        self.block_bytes = size * torch.empty((), dtype=dtype).element_size()
        if getattr(self, "resource_cache", None) is not None:
            # start() can be called again after stop(); release old terminal buffers first.
            self.resource_cache.release_host(f"{self.resource_prefix}/pinned")
            self.resource_cache.release_device(f"{self.resource_prefix}/slots")
            self.resource_cache.reserve_host(f"{self.resource_prefix}/pinned", self.ram_window * self.block_bytes)
            self.resource_cache.reserve_device(f"{self.resource_prefix}/slots", self.window * self.block_bytes)
        if self.resource_cache is not None and not self.source_names:
            for index, block in enumerate(self.blocks):
                name = f"{self.resource_prefix}/{index}"
                self.resource_cache.register_source(
                    name, (lambda index=index: (self.packed[index],)) if self.packed is not None
                          else (lambda block=block: tuple(p.detach() for p in block.parameters())),
                    self.block_bytes if self.packed is not None else
                    sum(p.numel() * p.element_size() for p in block.parameters()),
                )
                self.source_names.append(name)
        self.pinned = [self.rows.buffer() for _ in range(self.ram_window)] if self.rows is not None else \
            [torch.empty(size, dtype=dtype, pin_memory=True) for _ in range(self.ram_window)]
        self.requested = 0  # blocks whose copy load() has enqueued
        self.gpu = torch.empty((self.window, size), dtype=dtype, device=self.device)
        self.slots = [self.slot(row) if self.slot is not None else module_on(row, like=self.blocks[0])
                      for row in self.gpu]
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
        try:
            for i, block in enumerate(self.blocks if self.once else cycle(self.blocks)):
                buffer, copied = self.free.get()
                if buffer is None:
                    return
                if copied is not None:
                    copied.synchronize()  # the copy to the GPU has finished reading this buffer
                started = time.perf_counter()
                if self.rows is not None:
                    row = self.rows.read(i % len(self.blocks), buffer)
                else:
                    values = self.resource_cache.source(self.source_names[i % len(self.blocks)]) \
                        if self.resource_cache is not None else ((self.packed[i % len(self.blocks)],)
                        if self.packed is not None else (p.detach() for p in block.parameters()))
                    torch.cat([p.reshape(-1) for p in values], out=buffer)
                    row = buffer
                item = ((buffer, row), dict(block=i % len(self.blocks), pack_s=time.perf_counter() - started))
                self.filled.put(item if self.trace is not None else item[0])
        except Exception as error:
            self.filled.put(error)  # propagate instead of hanging the consumer forever

    def load(self, slot, after):
        """Stage 2: enqueues the copy of the next block from its pinned buffer into `slot`, after the event `after`."""
        if self.once and self.requested == len(self.blocks):
            return  # the pass is over: no block comes after the last
        self.requested += 1
        started = time.perf_counter()
        ready = not self.filled.empty()  # diagnostic only; get() remains authoritative
        buffer = self.filled.get()
        pack_wait_s = time.perf_counter() - started
        if isinstance(buffer, Exception):
            raise RuntimeError("Weight preparation failed") from buffer
        row = None
        if self.trace is not None:
            buffer, row = buffer
            row.update(pack_wait_s=pack_wait_s, pack_ready_at_request=ready)
        buffer, data = buffer  # the pinned buffer, and the row in it
        with torch.cuda.stream(self.copy_stream):
            if after is not None:
                self.copy_stream.wait_event(after)
            if row is not None:
                row['copy_begin'] = self.trace.event(self.copy_stream)
            self.gpu[slot].copy_(data, non_blocking=True)
            copied = torch.cuda.Event(enable_timing=row is not None)
            copied.record(self.copy_stream)
        self.free.put((buffer, copied))
        if row is not None:
            row['copied'] = copied
            self.trace.copies.append(row)
        self.loaded.append((slot, copied, row))

    def get_next(self):
        slot, copied, row = self.loaded.popleft()
        if self.trace is not None:
            self.trace.begin(row)
        else:
            torch.cuda.current_stream().wait_event(copied)
        return self.slots[slot]

    def release(self, block):
        if self.trace is not None:
            self.trace.end()
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        self.load(self.slots.index(block), after=done)

    def stop(self):
        if self.thread is not None:
            self.free.put((None, None))
            self.thread.join()
            self.thread = None

    def close(self):
        """Finish outstanding copies/compute and release all staging buffers.

        Unlike stop(), this is terminal. It also removes references held by queues
        and slot modules, which otherwise keep pinned RAM and VRAM alive.
        """
        self.stop()
        if getattr(self, "resource_cache", None) is not None:
            for name in self.source_names:
                self.resource_cache.unregister_source(name)
            self.source_names.clear()
            self.resource_cache.release_host(f"{self.resource_prefix}/pinned")
            self.resource_cache.release_device(f"{self.resource_prefix}/slots")
        if getattr(self, "gpu", None) is not None:
            torch.cuda.synchronize(self.device)
        if getattr(self, "rows", None) is not None:
            for buffer in getattr(self, "pinned", []):
                self.rows.unpin(buffer)
            self.rows.close()
            self.pinned = []  # a second close() must not unpin them again
        self.slots = []
        self.pinned = []
        self.gpu = None
        self.loaded = deque()
        self.free = None
        self.filled = None
        self.blocks = None
        self.packed = None
