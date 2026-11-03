"""Diagnostic-only blocking transfer measurements; not a prefetch deadline counter."""
from contextlib import ExitStack
import sys
import time
from unittest.mock import patch
import torch


class TransferProbe:
    def __init__(self):
        self.records = []

    def __enter__(self):
        self.stack = ExitStack()
        original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_

        def measure(source, call, destination_type, label):
            if destination_type == source.device.type or {destination_type, source.device.type} != {'cpu', 'cuda'}:
                return call()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            t = time.perf_counter()
            result = call()
            elapsed = time.perf_counter() - t
            end.record()
            self.records.append((source.numel() * source.element_size(), elapsed, start, end, label))
            return result

        def to(source, *args, **kwargs):
            target = kwargs.get('device', args[0] if args else None)
            if torch.is_tensor(target):
                target = target.device
            destination = torch.device(target).type if isinstance(target, (str, torch.device)) else source.device.type
            label = 'weights' if sys._getframe(1).f_code.co_name == 'hacked_get_attr' else 'cache_and_other'
            return measure(source, lambda: original_to(source, *args, **kwargs), destination, label)

        def copy(target, source, *args, **kwargs):
            return measure(source, lambda: original_copy(target, source, *args, **kwargs), target.device.type, 'cache_and_other')

        self.stack.enter_context(patch.object(torch.Tensor, 'to', to))
        self.stack.enter_context(patch.object(torch.Tensor, 'copy_', copy))
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.stack.close()
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.started

    def report(self):
        def aggregate(rows):
            return {'copies': len(rows), 'bytes': sum(r[0] for r in rows),
                    'host_blocked_seconds': sum(r[1] for r in rows),
                    'copy_stream_seconds': sum(r[2].elapsed_time(r[3]) for r in rows) / 1000}
        return {**aggregate(self.records), 'wall_seconds': self.seconds,
                'prefetch_enabled': False, 'deadline_misses': None,
                'notes': 'Host copy time includes waiting for producer kernels. Stream time can include CPU/page-fault gaps. Neither is pure PCIe time.',
                'groups': {label: aggregate([r for r in self.records if r[4] == label])
                           for label in sorted({r[4] for r in self.records})}}
