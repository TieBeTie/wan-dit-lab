import pytest

from wan_dit_lab.offloading.resource_schedule import Node, Resource, ResourcePlan, sign_pvalue, simulate


def test_shared_resource_acquired_once_released_after_last_use():
    events = []
    resource = Resource("state", lambda p: events.append("load"), lambda v: events.append("free"))
    nodes = [Node(str(i), ("state",), lambda p, i=i: events.append(i)) for i in range(3)]
    plan = ResourcePlan([resource], nodes)
    plan.run()
    assert events == ["load", 0, 1, 2, "free"]
    assert not plan.values


def test_failure_cleans_reverse_order_and_preserves_original_error():
    events = []

    def fail(value):
        events.append("free b")
        raise RuntimeError("cleanup")

    resources = [
        Resource("a", lambda p: 1, lambda v: events.append("free a")),
        Resource("b", lambda p: 2, fail),
    ]

    def node(plan):
        raise ValueError("node")

    plan = ResourcePlan(resources, [Node("fail", ("a", "b"), node)])
    with pytest.raises(ValueError, match="node"):
        plan.run()
    assert events == ["free b", "free a"]
    assert not plan.values


def test_cleanup_failure_is_reported_without_a_node_failure():
    def fail(value):
        raise RuntimeError("cleanup")

    plan = ResourcePlan([Resource("a", lambda p: 1, fail)], [Node("x", ("a",), lambda p: None)])
    with pytest.raises(RuntimeError, match="cleanup"):
        plan.run()


def test_invalid_plan_rejected_before_acquisition():
    with pytest.raises(ValueError, match="Unknown"):
        ResourcePlan([], [Node("x", ("unknown",), lambda p: None)])


def test_three_engine_overlap_and_independent_buffers():
    assert simulate([2] * 6, [3] * 6, [10] * 6, 2, 1) == 80
    assert simulate([2] * 6, [3] * 6, [10] * 6, 2, 2) == 65
    assert simulate([1] * 6, [20] * 6, [5] * 6, 2, 4) == 126
    with pytest.raises(ValueError):
        simulate([1], [1], [-1], 1, 1)


def test_exact_sign_test_and_ties():
    assert sign_pvalue(0, 0) == 1
    assert sign_pvalue(4, 0) == 0.125
    assert sign_pvalue(18, 6) < 0.025
