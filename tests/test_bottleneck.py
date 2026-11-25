"""bottleneck: the arithmetic and traffic of the kernels of a block, the verdict compute / bandwidth / launch, the
scaling class and the nvidia-smi summary behind docs/report-2026-11-16-bottlenecks.md."""

import math

import pytest

from wan_dit_lab.harness import bottleneck as bn


def test_gemm_launches_of_a_block_are_named_in_the_order_of_the_block():
    ms = [1.967, 0.657, 0.664, 0.662, 3.507, 3.010]
    assert bn.name_gemms(ms) == {"self QKV": 1.967, "self O": 0.657, "cross Q": 0.664, "cross O": 0.662,
                                 "FFN 1 (+GELU, quantize of FFN 2)": 3.507, "FFN 2": 3.010}


def test_a_block_with_another_number_of_gemms_is_refused():
    with pytest.raises(ValueError):
        bn.name_gemms([1.0, 2.0])


def test_w4a4_gemm_macs_and_bytes_of_one_launch():
    c = bn.w4a4_gemm(tokens=4080, k=3072, n=3072, rank=32)
    assert c.int4_macs == 4080 * 3072 * 3072  # the 4-bit product of the real rows
    assert c.lora_macs == 4080 * 32 * 3072  # the 16-bit low-rank branch of the output
    # read: padded INT4 activation, INT4 weights, weight and activation scales, low-rank activation and up projection
    read = 4096 * 3072 // 2 + 3072 * 3072 // 2 + 3072 * 3072 // 64 * 2 + 3072 // 64 * 4096 * 2 + 4096 * 32 * 4 + 3072 * 32 * 2
    assert c.read_bytes == read
    assert c.write_bytes == 4080 * 3072 * 2  # bfloat16 output


def test_ffn1_epilogue_writes_the_int4_hidden_and_its_scales():
    c = bn.w4a4_gemm(tokens=4080, k=3072, n=14336, rank=32, next_layer_epilogue=True)
    plain = bn.w4a4_gemm(tokens=4080, k=3072, n=14336, rank=32)
    # the hidden state leaves as INT4 with scales, not as bfloat16
    assert c.write_bytes == 4096 * 14336 // 2 + 14336 // 64 * 4096 * 2 + 4096 * 32 * 4
    assert c.write_bytes < plain.write_bytes


def test_quantize_launch_reads_bf16_and_writes_int4():
    c = bn.w4a4_quantize(tokens=4080, k=3072, rank=32)
    assert c.read_bytes == 4080 * 3072 * 2
    assert c.write_bytes == 4096 * 3072 // 2 + 3072 // 64 * 4096 * 2 + 4096 * 32 * 4
    assert c.lora_macs == 4080 * 3072 * 32


def test_sage_macs_count_qk_and_pv_over_all_keys():
    c = bn.sage(tokens=4080, keys=16320, heads=24, head_dim=128)
    assert c.int8_macs == 4080 * 16320 * 24 * 128 * 2  # Q K^T (INT8) and P V (FP8 at the same rate class)
    # Q INT8, K INT8, V FP8 read, bfloat16 output
    assert c.read_bytes == 4080 * 3072 + 16320 * 3072 * 2
    assert c.write_bytes == 4080 * 3072 * 2


def test_verdict_names_the_bound_that_is_closer_to_its_peak():
    assert bn.verdict(bw_share=0.10, ops_share=0.70, ms=5.0) == "compute bound"
    assert bn.verdict(bw_share=0.80, ops_share=0.10, ms=5.0) == "bandwidth bound"
    assert bn.verdict(bw_share=0.05, ops_share=0.04, ms=0.01) == "launch overhead"
    assert bn.verdict(bw_share=0.20, ops_share=0.25, ms=0.4).startswith("neither")


def test_scaling_exponent_tells_constant_linear_and_quadratic():
    xs = [0.5, 1.0, 2.0]
    assert bn.exponent(xs, [1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert bn.exponent(xs, [0.5, 1.0, 2.0]) == pytest.approx(1.0)
    assert bn.exponent(xs, [0.25, 1.0, 4.0]) == pytest.approx(2.0)
    assert bn.scaling_class(0.02) == "constant"
    assert bn.scaling_class(0.97) == "linear"
    assert bn.scaling_class(1.9) == "quadratic"
    assert bn.scaling_class(1.4) == "between linear and quadratic (1.40)"


def test_smi_summary_keeps_the_busy_samples_only():
    header = ["timestamp", "utilization.gpu [%]", "power.draw [W]", "clocks.current.sm [MHz]"]
    rows = [["t0", "5 %", "20.0 W", "1500 MHz"], ["t1", "99 %", "98.0 W", "2300 MHz"], ["t2", "100 %", "100.0 W", "2400 MHz"]]
    s = bn.smi_summary(header, rows, busy=90)
    assert s["samples"] == 2
    assert s["utilization.gpu"] == (99.0, 99.5, 100.0)
    assert s["clocks.current.sm"] == (2300.0, 2350.0, 2400.0)
    assert math.isclose(s["power.draw"][1], 99.0)
