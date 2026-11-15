"""Command-line entry points for sample generation and graph evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supplygraph.data import generate_synthetic_network, load_network, write_network
from supplygraph.model import evaluate
from supplygraph.reports import write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supplygraph",
        description="Evaluate synthetic supply-network disruption exposure with message passing.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a fictional network and scenarios")
    generate.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    generate.add_argument("--seed", type=int, default=61)

    evaluation = commands.add_parser("evaluate", help="evaluate exposure on disruption scenarios")
    evaluation.add_argument("--nodes", type=Path, default=Path("data/sample/nodes.csv"))
    evaluation.add_argument("--edges", type=Path, default=Path("data/sample/edges.csv"))
    evaluation.add_argument(
        "--disruptions",
        type=Path,
        default=Path("data/sample/disruptions.csv"),
    )
    evaluation.add_argument("--output-dir", type=Path, default=Path("reports"))
    evaluation.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            network = generate_synthetic_network(seed=args.seed)
            write_network(network, args.output_dir)
            print(
                f"Wrote {len(network.nodes)} nodes, {len(network.edges)} edges, "
                f"and {len(network.disruptions)} scenarios to {args.output_dir}"
            )
            print("These fictional records are for software demonstration, not real supply data.")
            return 0

        network = load_network(args.nodes, args.edges, args.disruptions)
        metrics, rows = evaluate(network, positive_threshold=args.threshold)
        metrics_path, exposure_path = write_reports(metrics.to_dict(), rows, args.output_dir)
    except (OSError, TypeError, ValueError) as exc:
        print(f"supplygraph: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics.to_dict(), indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Scenario exposure report: {exposure_path}")
    print(
        "Interpretation: exposure flags are synthetic planning signals for human review; "
        "they do not decide resource allocation or measure real community risk."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
