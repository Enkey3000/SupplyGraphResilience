"""Synthetic network generation and tabular input validation."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

NODE_TYPES = ("supplier", "warehouse", "recipient")


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    capacity_units: float
    lead_time_days: float
    latitude: float
    longitude: float
    community_index: int


@dataclass(frozen=True)
class Edge:
    source_id: str
    target_id: str
    weekly_units: float
    transit_days: float


@dataclass(frozen=True)
class Disruption:
    disruption_id: str
    node_id: str
    severity: float


@dataclass(frozen=True)
class Network:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    disruptions: tuple[Disruption, ...]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_synthetic_network(seed: int = 61) -> Network:
    """Create a small fictional supplier/warehouse/recipient network."""
    rng = random.Random(seed)
    nodes: list[Node] = []
    for index in range(8):
        nodes.append(
            Node(
                node_id=f"supplier-{index + 1:02d}",
                node_type="supplier",
                capacity_units=float(rng.randint(80, 180)),
                lead_time_days=round(rng.uniform(1, 5), 2),
                latitude=round(39.2 + rng.random() * 1.8, 5),
                longitude=round(-77.5 + rng.random() * 2.2, 5),
                community_index=index % 4,
            )
        )
    for index in range(4):
        nodes.append(
            Node(
                node_id=f"warehouse-{index + 1:02d}",
                node_type="warehouse",
                capacity_units=float(rng.randint(200, 400)),
                lead_time_days=round(rng.uniform(0.5, 2), 2),
                latitude=round(39.2 + rng.random() * 1.8, 5),
                longitude=round(-77.5 + rng.random() * 2.2, 5),
                community_index=index,
            )
        )
    for index in range(12):
        nodes.append(
            Node(
                node_id=f"recipient-{index + 1:02d}",
                node_type="recipient",
                capacity_units=float(rng.randint(25, 70)),
                lead_time_days=round(rng.uniform(0.2, 1.2), 2),
                latitude=round(39.2 + rng.random() * 1.8, 5),
                longitude=round(-77.5 + rng.random() * 2.2, 5),
                community_index=index % 4,
            )
        )

    edges: list[Edge] = []
    suppliers = [node for node in nodes if node.node_type == "supplier"]
    warehouses = [node for node in nodes if node.node_type == "warehouse"]
    recipients = [node for node in nodes if node.node_type == "recipient"]
    for warehouse_index, warehouse in enumerate(warehouses):
        for supplier in suppliers:
            if rng.random() < 0.6 or supplier == suppliers[warehouse_index]:
                edges.append(
                    Edge(
                        source_id=supplier.node_id,
                        target_id=warehouse.node_id,
                        weekly_units=float(rng.randint(20, 85)),
                        transit_days=round(rng.uniform(1, 4), 2),
                    )
                )
    for index, recipient in enumerate(recipients):
        primary = warehouses[index % len(warehouses)]
        edges.append(
            Edge(
                source_id=primary.node_id,
                target_id=recipient.node_id,
                weekly_units=float(rng.randint(15, 50)),
                transit_days=round(rng.uniform(0.3, 1.5), 2),
            )
        )
        if index % 3 == 0:
            backup = warehouses[(index + 1) % len(warehouses)]
            edges.append(
                Edge(
                    source_id=backup.node_id,
                    target_id=recipient.node_id,
                    weekly_units=float(rng.randint(5, 20)),
                    transit_days=round(rng.uniform(0.5, 2), 2),
                )
            )
    disruptions = tuple(
        Disruption(
            disruption_id=f"scenario-{index + 1:02d}",
            node_id=supplier.node_id,
            severity=1.0,
        )
        for index, supplier in enumerate(suppliers[:6])
    )
    return Network(tuple(nodes), tuple(edges), disruptions)


def write_network(network: Network, output_dir: Path) -> None:
    _write_csv(
        output_dir / "nodes.csv",
        (
            "node_id",
            "node_type",
            "capacity_units",
            "lead_time_days",
            "latitude",
            "longitude",
            "community_index",
        ),
        [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "capacity_units": node.capacity_units,
                "lead_time_days": node.lead_time_days,
                "latitude": node.latitude,
                "longitude": node.longitude,
                "community_index": node.community_index,
            }
            for node in network.nodes
        ],
    )
    _write_csv(
        output_dir / "edges.csv",
        ("source_id", "target_id", "weekly_units", "transit_days"),
        [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "weekly_units": edge.weekly_units,
                "transit_days": edge.transit_days,
            }
            for edge in network.edges
        ],
    )
    _write_csv(
        output_dir / "disruptions.csv",
        ("disruption_id", "node_id", "severity"),
        [
            {
                "disruption_id": disruption.disruption_id,
                "node_id": disruption.node_id,
                "severity": disruption.severity,
            }
            for disruption in network.disruptions
        ],
    )


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def _number(row: dict[str, str], field: str, *, minimum: float | None = None) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{field} is out of range")
    return value


def load_network(nodes_path: Path, edges_path: Path, disruptions_path: Path) -> Network:
    nodes: list[Node] = []
    node_ids: set[str] = set()
    for row in _read_csv(
        nodes_path,
        {
            "node_id",
            "node_type",
            "capacity_units",
            "lead_time_days",
            "latitude",
            "longitude",
            "community_index",
        },
    ):
        node_id = row["node_id"].strip()
        node_type = row["node_type"].strip()
        if not node_id or node_id in node_ids:
            raise ValueError(f"Empty or duplicate node_id: {node_id!r}")
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unsupported node_type for {node_id}: {node_type!r}")
        latitude = _number(row, "latitude")
        longitude = _number(row, "longitude")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"Invalid coordinates for {node_id}")
        try:
            community_index = int(row["community_index"])
        except ValueError as exc:
            raise ValueError(f"community_index must be an integer for {node_id}") from exc
        if community_index < 0:
            raise ValueError(f"community_index must be non-negative for {node_id}")
        nodes.append(
            Node(
                node_id=node_id,
                node_type=node_type,
                capacity_units=_number(row, "capacity_units", minimum=0),
                lead_time_days=_number(row, "lead_time_days", minimum=0),
                latitude=latitude,
                longitude=longitude,
                community_index=community_index,
            )
        )
        node_ids.add(node_id)
    if not nodes:
        raise ValueError("Nodes CSV contains no nodes")

    edges: list[Edge] = []
    seen_edges: set[tuple[str, str]] = set()
    for row in _read_csv(edges_path, {"source_id", "target_id", "weekly_units", "transit_days"}):
        source_id = row["source_id"].strip()
        target_id = row["target_id"].strip()
        key = (source_id, target_id)
        if source_id not in node_ids or target_id not in node_ids:
            raise ValueError(f"Edge refers to unknown node: {key}")
        if key in seen_edges:
            raise ValueError(f"Duplicate directed edge: {key}")
        edges.append(
            Edge(
                source_id=source_id,
                target_id=target_id,
                weekly_units=_number(row, "weekly_units", minimum=0),
                transit_days=_number(row, "transit_days", minimum=0),
            )
        )
        seen_edges.add(key)
    if not edges:
        raise ValueError("Edges CSV contains no edges")

    disruptions: list[Disruption] = []
    seen_scenarios: set[str] = set()
    for row in _read_csv(disruptions_path, {"disruption_id", "node_id", "severity"}):
        disruption_id = row["disruption_id"].strip()
        node_id = row["node_id"].strip()
        severity = _number(row, "severity", minimum=0)
        if not disruption_id or disruption_id in seen_scenarios:
            raise ValueError(f"Empty or duplicate disruption_id: {disruption_id!r}")
        if node_id not in node_ids:
            raise ValueError(f"Disruption refers to unknown node: {node_id}")
        if severity > 1:
            raise ValueError(f"Disruption severity must be between 0 and 1: {disruption_id}")
        disruptions.append(Disruption(disruption_id, node_id, severity))
        seen_scenarios.add(disruption_id)
    if not disruptions:
        raise ValueError("Disruptions CSV contains no scenarios")
    return Network(tuple(nodes), tuple(edges), tuple(disruptions))
