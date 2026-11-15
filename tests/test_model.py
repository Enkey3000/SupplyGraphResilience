import csv

import pytest

from supplygraph.data import generate_synthetic_network
from supplygraph.model import evaluate, graph_baseline_scores
from supplygraph.reports import write_reports


def test_graph_evaluation_reports_all_scenario_node_pairs():
    network = generate_synthetic_network()
    metrics, rows = evaluate(network)
    assert len(rows) == len(network.disruptions) * len(network.nodes)
    assert metrics.disruption_scenarios == len(network.disruptions)
    assert metrics.evaluated_nodes == len(rows)
    assert set(metrics.class_recall) == {"not_exposed", "exposed"}
    assert len(metrics.confusion_matrix) == 2


def test_simple_graph_baseline_scores_are_finite_and_shaped():
    network = generate_synthetic_network()
    scores = graph_baseline_scores(network)
    assert scores.shape == (len(network.disruptions), len(network.nodes))
    assert scores.min() >= 0
    assert scores.max() <= 1


def test_model_outputs_human_review_statuses():
    network = generate_synthetic_network()
    _, rows = evaluate(network)
    assert {row["review_status"] for row in rows} <= {
        "review_exposure",
        "no_flag_review_sample",
    }


def test_evaluation_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        evaluate(generate_synthetic_network(), positive_threshold=1.5)


def test_reports_are_written(tmp_path):
    network = generate_synthetic_network()
    metrics, rows = evaluate(network)
    metrics_path, rows_path = write_reports(metrics.to_dict(), rows, tmp_path)
    assert metrics_path.is_file()
    with rows_path.open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == len(rows)
