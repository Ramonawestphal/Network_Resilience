import pytest

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds


def test_compute_scaled_max_rounds_disabled_returns_reference():
    assert compute_scaled_max_rounds(20, num_nodes=100, reference_n=40, enabled=False) == 20


def test_compute_scaled_max_rounds_linear_at_reference_n():
    assert compute_scaled_max_rounds(20, num_nodes=40, reference_n=40, enabled=True) == 20


def test_compute_scaled_max_rounds_scales_up_for_large_graph():
    assert compute_scaled_max_rounds(20, num_nodes=80, reference_n=40, enabled=True) == 40


def test_compute_scaled_max_rounds_matches_budget_scaling_shape():
    n, ref_n, ref_b, ref_mr = 100, 40, 3, 20
    b = compute_scaled_budget(ref_b, num_nodes=n, reference_n=ref_n, enabled=True)
    mr = compute_scaled_max_rounds(ref_mr, num_nodes=n, reference_n=ref_n, enabled=True)
    assert b == round(ref_b * n / ref_n)
    assert mr == round(ref_mr * n / ref_n)


def test_compute_scaled_max_rounds_requires_positive_reference():
    with pytest.raises(ValueError, match="reference_max_rounds"):
        compute_scaled_max_rounds(0, num_nodes=10, reference_n=40, enabled=True)
