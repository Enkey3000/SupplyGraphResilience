# SupplyGraph Resilience

**Explore how a disruption at one supplier could ripple through a fictional
distribution network—and expose the assumptions behind the score.**

SupplyGraph Resilience is a graph-learning portfolio prototype for screening
downstream supply exposure. It models supplier, warehouse, and recipient nodes;
directed flow edges; and synthetic supplier-disruption scenarios. A small message-
passing neural model is compared with a node-and-flow baseline and emits recipient
exposure flags for human review.

> **Scope:** All sample nodes, connections, scenarios, and labels are synthetic. The
> model does not forecast real shortages, assess actual communities, recommend
> allocation, or make operational decisions.

## Executive summary and case study

A fictional regional food network wants to understand which recipient sites might be
affected if a supplier is disrupted. The prototype generates a transparent toy network,
simulates disruptions, evaluates a graph model against a simpler baseline, and exports
scenario-level results so planners can inspect the graph assumptions. It is an
engineering demonstration—not a validated operational risk assessment.

## Quick start (PowerShell)

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
supplygraph evaluate `
  --nodes data/sample/nodes.csv `
  --edges data/sample/edges.csv `
  --disruptions data/sample/disruptions.csv `
  --output-dir build/reports
python -m pytest
ruff check .
```

Run without installing the CLI:

```powershell
$env:PYTHONPATH = "src"
python -m supplygraph.cli evaluate --nodes data/sample/nodes.csv --edges data/sample/edges.csv --disruptions data/sample/disruptions.csv --output-dir build/reports
```

Regenerate the deterministic synthetic network:

```powershell
supplygraph generate --output-dir data/sample --seed 61
```

Evaluation writes `build/reports/evaluation.json` and
`build/reports/scenario_exposure.csv`. Every exposure row includes a scenario ID,
disrupted supplier, node type, community index, simulated label, predicted label, and
score.

## Data contract

The project uses three CSV tables:

- `nodes.csv`: node ID/type, capacity, lead time, coordinates, and synthetic community
  index.
- `edges.csv`: directed source/target, weekly units, and transit days.
- `disruptions.csv`: scenario ID, disrupted node ID, and severity in `[0, 1]`.

The loader rejects duplicate IDs and edges, unknown endpoints, invalid node types,
out-of-range coordinates, and non-finite or negative numeric values.

## Five assumptions and how the project addresses them

1. **Network links reflect supply relationships.** Keep directed edges explicit and
   validate endpoint IDs.
2. **Flow and capacity affect disruption propagation.** Include edge volume and node
   capacity in the scenario simulation and model features.
3. **A disruption can affect downstream nodes.** Use directed message passing and a
   simple baseline for comparison.
4. **Risk can vary by recipient area.** Report recipient-level rows and community
   indices, explicitly marked as synthetic in this demo.
5. **A score should inform, not automate allocation.** Export flags for planner review;
   never decide resource distribution.

## Foreseeable questions

**What is the graph neural network?** It builds node representations by aggregating
messages from connected nodes and then propagates a disruption-seeded signal.

**What is the baseline?** A logistic classifier using node attributes and aggregate
in/out flow totals without disruption message passing.

**How do you prevent a scenario from leaking into its own evaluation?** The evaluation
holds out one disruption scenario at a time for scoring.

**Are the exposure labels real?** No. They are generated from a documented toy
flow-threshold rule.

**Can this tell a food bank how to allocate resources?** No. It is a synthetic
screening example, not an allocation recommendation or real community-risk assessment.

## Automation, optimization, and potential savings

- **Automation:** Recompute scenario exposure summaries when an authorized network
  snapshot changes.
- **Optimization:** Help planners inspect supplier concentration and explore backup
  links or contingency scenarios.
- **Potential cost savings:** Better contingency planning might reduce disruption
  response time or spoilage, but this project claims no achieved savings. Any estimate
  requires validated operational data and a reviewed pilot.

## Project layout

```text
SupplyGraphResilience/
├── .github/workflows/ci.yml
├── data/sample/                 # Synthetic node, edge, and disruption CSVs
├── docs/methodology.md
├── reports/                     # Generated evaluation artifacts
├── src/supplygraph/              # Graph data, model, reports, CLI
├── tests/
├── pyproject.toml
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).
