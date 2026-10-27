"""Count actual CUDA graph kernel nodes when CUPTI drops its device timeline."""

import ctypes
import sys

import torch


def graph_kernels(fn):
    driver = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
              else ctypes.CDLL("libcuda.so.1"))
    pointer = ctypes.c_void_p
    signatures = {
        "cuGraphGetNodes": [pointer, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)],
        "cuGraphNodeGetType": [pointer, ctypes.POINTER(ctypes.c_int)],
        "cuGraphChildGraphNodeGetGraph": [pointer, ctypes.POINTER(pointer)],
    }
    for name, args in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = args, ctypes.c_int

    def call(name, *args):
        code = getattr(driver, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed with CUDA error {code}")

    def visit(handle):
        size = ctypes.c_size_t()
        call("cuGraphGetNodes", handle, None, ctypes.byref(size))
        nodes = (pointer * size.value)()
        call("cuGraphGetNodes", handle, nodes, ctypes.byref(size))
        names = []
        for node in nodes[:size.value]:
            kind = ctypes.c_int()
            call("cuGraphNodeGetType", node, ctypes.byref(kind))
            if kind.value == 0:  # CU_GRAPH_NODE_TYPE_KERNEL
                names.append("CUDA kernel node")
            elif kind.value == 4:  # CU_GRAPH_NODE_TYPE_GRAPH
                child = pointer()
                call("cuGraphChildGraphNodeGetGraph", node, ctypes.byref(child))
                names.extend(visit(child))
        return names

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        _output = fn()  # Keep captured outputs alive until the graph has been inspected.
    names = visit(graph.raw_cuda_graph())
    if not names:
        raise RuntimeError("Captured CUDA graph contains no kernel nodes")
    return names
