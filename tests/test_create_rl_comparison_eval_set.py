from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    ("argv", "expected_message"),
    [
        (["prog", "--num-graphs", "0"], "--num-graphs must be a positive integer."),
        (
            ["prog", "--seeds-per-graph", "-1"],
            "--seeds-per-graph must be a positive integer.",
        ),
    ],
)
def test_parse_args_rejects_non_positive_count_arguments(
    monkeypatch, capsys, argv, expected_message
):
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        create_rl_comparison_eval_set._parse_args()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert expected_message in captured.err
