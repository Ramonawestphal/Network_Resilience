"""Tests for graph loading utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import cascading_rl.graph.generation as gen


def test_load_real_world_graph_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown real-world graph"):
        gen.load_real_world_graph("not_a_dataset")


def test_load_real_world_graph_reads_triangle(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    csv_path = processed / "ieee300_edges.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from", "to"])
        w.writerow([0, 1])
        w.writerow([1, 2])
        w.writerow([0, 2])

    g = gen.load_real_world_graph("ieee300", data_dir=processed)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert g.graph.get("name") == "ieee300"
