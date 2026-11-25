from wan_dit_lab.harness.gemm_tile.bench import LAYERS, forecast


def test_forecast_scales_the_gemm_share_of_a_chunk_by_the_measured_ratio():
    # GEMM 927.1 ms of a 2564.5 ms chunk; a tiling 10% faster on every layer saves 92.7 ms of the chunk
    base = {name: 1.0 for name in LAYERS}
    variant = {name: 0.9 for name in LAYERS}
    f = forecast(base, variant, gemm_ms=927.1, chunk_ms=2564.5)
    assert abs(f["chunk_ms"] - (2564.5 - 92.71)) < 1e-6
    assert abs(f["gain"] - 2564.5 / (2564.5 - 92.71) + 1) < 1e-9


def test_forecast_weights_layers_by_their_time_in_the_block():
    # a layer that takes most of the base time dominates the forecast
    base = {name: 0.0 for name in LAYERS}
    base["ffn1"], base["qkv"] = 3.0, 1.0
    variant = dict(base, ffn1=1.5)
    f = forecast(base, variant, gemm_ms=400.0, chunk_ms=1000.0)
    assert abs(f["gemm_ms"] - 400.0 * 2.5 / 4.0) < 1e-9


def test_layers_have_the_shapes_of_the_working_point():
    # 4080 tokens padded to the 256 rows of a tile, dim 3072, feed forward 14336
    assert LAYERS["qkv"] == (4096, 9216, 3072, 1)
    assert LAYERS["ffn2"] == (4096, 3072, 14336, 1)
    assert sum(calls for *_, calls in LAYERS.values()) == 6
