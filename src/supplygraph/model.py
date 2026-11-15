"""Graph message-passing model and transparent graph baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score

from supplygraph.data import NODE_TYPES, Network


@dataclass(frozen=True)
class Evaluation:
    gnn_accuracy: float
    baseline_accuracy: float
    gnn_balanced_accuracy: float
    baseline_balanced_accuracy: float
    disruption_scenarios: int
    evaluated_nodes: int
    class_recall: dict[str, float]
    confusion_matrix: list[list[int]]
    positive_prediction_threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _prepare(network: Network) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    node_ids = [node.node_id for node in network.nodes]
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    features = np.asarray(
        [
            [
                node.capacity_units / 400.0,
                node.lead_time_days / 7.0,
                node.latitude / 90.0,
                node.longitude / 180.0,
                *[float(node.node_type == kind) for kind in NODE_TYPES],
            ]
            for node in network.nodes
        ],
        dtype=np.float64,
    )
    edges = np.asarray(
        [
            (index[edge.source_id], index[edge.target_id], edge.weekly_units / 100.0)
            for edge in network.edges
        ],
        dtype=np.float64,
    )
    labels = np.zeros((len(network.disruptions), len(network.nodes)), dtype=np.int64)
    for scenario_index, disruption in enumerate(network.disruptions):
        exposed = _reachable_recipients(network, disruption.node_id, disruption.severity)
        for node_id in exposed:
            labels[scenario_index, index[node_id]] = 1
    return node_ids, features, edges, labels


def _reachable_recipients(network: Network, disrupted_node: str, severity: float) -> set[str]:
    """Simulate directed downstream exposure from a reduced-capacity supply node."""
    node_by_id = {node.node_id: node for node in network.nodes}
    outgoing: dict[str, list[tuple[str, float]]] = {}
    for edge in network.edges:
        outgoing.setdefault(edge.source_id, []).append((edge.target_id, edge.weekly_units))
    supplier = node_by_id[disrupted_node]
    if supplier.node_type != "supplier" or severity <= 0:
        return set()
    capacity_fraction = min(1.0, supplier.capacity_units / 180.0)
    exposure_threshold = 22.0
    first_hop: dict[str, float] = {}
    for warehouse_id, flow in outgoing.get(disrupted_node, []):
        first_hop[warehouse_id] = first_hop.get(warehouse_id, 0.0) + flow * severity
    exposed: set[str] = set()
    for warehouse_id, impacted_flow in first_hop.items():
        for recipient_id, downstream_flow in outgoing.get(warehouse_id, []):
            proportional_impact = min(
                downstream_flow,
                impacted_flow * capacity_fraction,
            )
            if proportional_impact >= exposure_threshold:
                exposed.add(recipient_id)
    return exposed


def graph_baseline_scores(network: Network) -> np.ndarray:
    """Return a simple flow-weighted downstream reach score per disrupted supplier."""
    scores = np.zeros((len(network.disruptions), len(network.nodes)), dtype=np.float64)
    node_index = {node.node_id: i for i, node in enumerate(network.nodes)}
    for scenario_index, disruption in enumerate(network.disruptions):
        for node_id in _reachable_recipients(network, disruption.node_id, disruption.severity):
            scores[scenario_index, node_index[node_id]] = 1.0
    return scores


def gnn_scores(network: Network) -> np.ndarray:
    """Compute one-pass normalized graph message-passing scores per disruption scenario."""
    node_ids, node_features, edges, _ = _prepare(network)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    incoming_messages = np.zeros_like(node_features)
    incoming_weights = np.zeros(len(node_ids), dtype=np.float64)
    for source, target, weight in edges:
        incoming_messages[int(target)] += node_features[int(source)] * weight
        incoming_weights[int(target)] += weight
    nonzero = incoming_weights > 0
    incoming_messages[nonzero] /= incoming_weights[nonzero, None]
    embedded = np.tanh(node_features + incoming_messages)
    result = np.zeros((len(network.disruptions), len(node_ids)), dtype=np.float64)
    for scenario_index, disruption in enumerate(network.disruptions):
        disruption_vector = np.zeros(len(node_ids), dtype=np.float64)
        disruption_vector[node_index[disruption.node_id]] = disruption.severity
        messages = np.zeros_like(embedded)
        message_weights = np.zeros(len(node_ids), dtype=np.float64)
        for source, target, weight in edges:
            messages[int(target)] += embedded[int(source)] * weight * disruption_vector[int(source)]
            message_weights[int(target)] += weight * disruption_vector[int(source)]
        active = message_weights > 0
        messages[active] /= message_weights[active, None]
        propagated = np.tanh(embedded + messages)
        # A fixed, interpretable readout is trained below using generated labels;
        # this function returns a graph feature score as a stable baseline input.
        result[scenario_index] = np.linalg.norm(propagated - embedded, axis=1)
    return result


def _flat_features(
    network: Network,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    node_ids, node_features, edges, labels = _prepare(network)
    structural = np.zeros((len(node_ids), 2), dtype=np.float64)
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    for source, target, weight in edges:
        structural[int(source), 0] += weight
        structural[int(target), 1] += weight
    graph_features = np.concatenate((node_features, structural), axis=1)
    embeddings = np.zeros_like(node_features)
    incoming = np.zeros(len(node_ids), dtype=np.float64)
    for source, target, weight in edges:
        embeddings[int(target)] += node_features[int(source)] * weight
        incoming[int(target)] += weight
    active = incoming > 0
    embeddings[active] /= incoming[active, None]
    embeddings = np.tanh(node_features + embeddings)
    baseline_features: list[np.ndarray] = []
    gnn_features: list[np.ndarray] = []
    scenario_labels: list[np.ndarray] = []
    for scenario_index, disruption in enumerate(network.disruptions):
        seed_vector = np.zeros(len(node_ids), dtype=np.float64)
        seed_vector[index[disruption.node_id]] = disruption.severity
        messages = np.zeros_like(embeddings)
        weights = np.zeros(len(node_ids), dtype=np.float64)
        for source, target, edge_weight in edges:
            messages[int(target)] += (
                embeddings[int(source)] * edge_weight * seed_vector[int(source)]
            )
            weights[int(target)] += edge_weight * seed_vector[int(source)]
        active_messages = weights > 0
        messages[active_messages] /= weights[active_messages, None]
        propagated = np.tanh(embeddings + messages)
        scenario_metadata = np.full((len(node_ids), 1), disruption.severity)
        baseline_features.append(np.concatenate((graph_features, scenario_metadata), axis=1))
        gnn_features.append(
            np.concatenate(
                (
                    graph_features,
                    propagated - embeddings,
                    np.full((len(node_ids), 1), disruption.severity),
                ),
                axis=1,
            )
        )
        scenario_labels.append(labels[scenario_index])
    return (
        np.vstack(baseline_features),
        np.vstack(gnn_features),
        np.concatenate(scenario_labels),
        node_ids,
        labels,
    )


def evaluate(
    network: Network,
    *,
    positive_threshold: float = 0.5,
) -> tuple[Evaluation, list[dict[str, object]]]:
    """Train and evaluate a graph message-passing classifier on generated scenarios."""
    if not 0 <= positive_threshold <= 1:
        raise ValueError("positive_threshold must be between 0 and 1")
    baseline_x, gnn_x, y, node_ids, scenario_labels = _flat_features(network)
    scenario_count, node_count = scenario_labels.shape
    if scenario_count < 2:
        raise ValueError("At least two disruption scenarios are required")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Synthetic scenarios must contain both exposed and unexposed nodes")

    # Leave-one-disruption-scenario-out probabilities provide a small transparent
    # scenario-held-out evaluation without training and scoring on the same event.
    gnn_probabilities = np.zeros(len(y), dtype=np.float64)
    baseline_probabilities = np.zeros(len(y), dtype=np.float64)
    for held_out in range(scenario_count):
        train_mask = np.ones(scenario_count, dtype=bool)
        train_mask[held_out] = False
        train_rows = np.concatenate(
            [
                np.arange(index * node_count, (index + 1) * node_count)
                for index in range(scenario_count)
                if train_mask[index]
            ]
        )
        test_rows = np.arange(held_out * node_count, (held_out + 1) * node_count)
        gnn_classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=23,
        )
        gnn_classifier.fit(gnn_x[train_rows], y[train_rows])
        gnn_probabilities[test_rows] = gnn_classifier.predict_proba(gnn_x[test_rows])[:, 1]
        baseline_classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=23,
        )
        baseline_classifier.fit(baseline_x[train_rows], y[train_rows])
        baseline_probabilities[test_rows] = baseline_classifier.predict_proba(
            baseline_x[test_rows]
        )[:, 1]

    prediction = (gnn_probabilities >= positive_threshold).astype(int)
    baseline_prediction = (baseline_probabilities >= positive_threshold).astype(int)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    recall_values = recall_score(y, prediction, labels=[0, 1], average=None, zero_division=0)
    reports: list[dict[str, object]] = []
    for scenario_index, disruption in enumerate(network.disruptions):
        start = scenario_index * node_count
        for node_index, node_id in enumerate(node_ids):
            node = network.nodes[node_index]
            reports.append(
                {
                    "disruption_id": disruption.disruption_id,
                    "disrupted_node_id": disruption.node_id,
                    "node_id": node_id,
                    "node_type": node.node_type,
                    "community_index": node.community_index,
                    "actual_exposure": int(scenario_labels[scenario_index, node_index]),
                    "predicted_exposure": int(prediction[start + node_index]),
                    "exposure_score": round(float(gnn_probabilities[start + node_index]), 4),
                    "review_status": (
                        "review_exposure"
                        if prediction[start + node_index] == 1
                        else "no_flag_review_sample"
                    ),
                }
            )
    metrics = Evaluation(
        gnn_accuracy=round(float(accuracy_score(y, prediction)), 4),
        baseline_accuracy=round(float(accuracy_score(y, baseline_prediction)), 4),
        gnn_balanced_accuracy=round(float(balanced_accuracy_score(y, prediction)), 4),
        baseline_balanced_accuracy=round(float(balanced_accuracy_score(y, baseline_prediction)), 4),
        disruption_scenarios=scenario_count,
        evaluated_nodes=len(y),
        class_recall={
            "not_exposed": round(float(recall_values[0]), 4),
            "exposed": round(float(recall_values[1]), 4),
        },
        confusion_matrix=matrix.tolist(),
        positive_prediction_threshold=positive_threshold,
    )
    return metrics, reports
