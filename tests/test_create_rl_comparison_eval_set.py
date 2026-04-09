from __future__ import annotations

from scripts import create_rl_comparison_eval_set


def test_resolve_budget_scaling_defaults_to_training_graph_upper_bound():
    config = {
        "training": {
            "graph": {"n_range": [24, 36]},
        }
    }

    n_ref, scale_budget = create_rl_comparison_eval_set._resolve_budget_scaling(config)

    assert n_ref == 36
    assert scale_budget is False


def test_resolve_budget_scaling_prefers_explicit_shared_budget_scaling_config():
    config = {
        "training": {
            "graph": {"n_range": [24, 36]},
        },
        "budget_scaling": {
            "enabled": True,
            "reference_n": 52,
        },
    }

    n_ref, scale_budget = create_rl_comparison_eval_set._resolve_budget_scaling(config)

    assert n_ref == 52
    assert scale_budget is True
