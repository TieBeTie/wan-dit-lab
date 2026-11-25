from wan_dit_lab.harness.metric_sensitivity import CONFIGS, FRAMES, clip_frames


def test_every_bad_configuration_has_a_good_counterpart_with_the_same_seeds_and_scenes():
    bad = {k: v for k, v in CONFIGS.items() if v["good"] is not None}
    assert set(bad) == {"laststep608", "nointerp512", "uniform512", "win16_608", "win16_704"}
    for name, c in bad.items():
        g = CONFIGS[c["good"]]
        assert set(c["seeds"]) <= set(g["seeds"]) and set(c["scenes"]) <= set(g["scenes"]), name


def test_videos_are_cut_to_the_three_chunks_of_the_reference():
    assert FRAMES == 93
    assert len(clip_frames(list(range(189)))) == 93 and len(clip_frames(list(range(93)))) == 93
