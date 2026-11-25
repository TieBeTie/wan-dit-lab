"""torch 2.10 caches the C++ declarations of size arrays by id(writeline). A bound method like `lines.append` is a new
object on each access, and once freed its id is taken by the next one: a writeline of the second pass of memory
planning then reuses a declaration written by the first, which lands later in the file. On 2026-11-23 block_2 of
544x960 failed so ("int_array_14 undeclared identifier", declared 226 lines below its use); 16.11 the ids happened
not to collide and the same code compiled."""
import itertools

import pytest

from wan_dit_lab.runtime import aoti_patches

cpp_wrapper_cpu = pytest.importorskip("torch._inductor.codegen.cpp_wrapper_cpu")


class Wrapper:
    codegen_int_array_var = cpp_wrapper_cpu.CppWrapperCpu.codegen_int_array_var

    def __init__(self):
        self.codegen_int_array_var_cache = {}
        self.int_array_id = itertools.count()

    def _codegen_int_array_var_impl(self, int_array, writeline, known_statically):
        var = f"int_array_{next(self.int_array_id)}"
        writeline(f"static constexpr int64_t {var}[] = {int_array};")
        return var


def declared_where_used(wrapper):
    """Each of many short-lived writelines gets its declaration written into its own lines."""
    for _ in range(200):
        lines = []
        var = wrapper.codegen_int_array_var("{4080LL, 3072LL}", lines.append)
        if not any(var in line for line in lines):
            return False
    return True


def test_the_unpatched_cache_reuses_a_declaration_of_another_writeline():
    assert not declared_where_used(Wrapper()), "torch fixed the cache: drop aoti_patches"


def test_the_patched_cache_declares_each_array_where_it_is_used():
    aoti_patches.apply(Wrapper)
    assert declared_where_used(Wrapper())
    wrapper, lines = Wrapper(), []
    first = wrapper.codegen_int_array_var("{1LL}", lines.append)
    assert wrapper.codegen_int_array_var("{1LL}", lines.append) == first, "the same writeline still shares one"
    assert len(lines) == 1
