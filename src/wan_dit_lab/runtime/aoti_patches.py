"""Fixes of torch's AOTInductor applied by the export (scripts/conversion/export_live.py) before it compiles."""


def apply(wrapper_class=None):
    """Size arrays cached by the writeline itself instead of id(writeline) (tests/test_aoti_int_arrays.py): the key
    holds the bound method, so its id cannot pass to another, and two bound methods of one target compare equal."""
    if wrapper_class is None:
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu as wrapper_class
    if getattr(wrapper_class.codegen_int_array_var, "patched", False):
        return

    def codegen_int_array_var(self, int_array, writeline, known_statically=False, graph=None):
        key = ("by-writeline", int_array, writeline, known_statically, id(graph) if graph else None)
        cache = self.codegen_int_array_var_cache
        if key not in cache:
            cache[key] = self._codegen_int_array_var_impl(int_array, writeline, known_statically)
        return cache[key]

    codegen_int_array_var.patched = True
    wrapper_class.codegen_int_array_var = codegen_int_array_var
