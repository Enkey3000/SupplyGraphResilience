# Graph modeling methodology

## Synthetic case study

SupplyGraph Resilience represents a fictional food-distribution network as directed
nodes and edges. Node types are suppliers, warehouses, and recipient sites. Edges
carry weekly volume and transit-time attributes. A disruption scenario reduces one
supplier's capacity and generates downstream recipient exposure labels using a
transparent flow-threshold simulation. All input data and labels are synthetic.

## Model and baseline

Node features include scaled capacity, lead time, coordinates, and node-type indicators.
The graph model computes a normalized incoming message-passing representation, then
propagates a disruption-seeded message through directed supplier-to-warehouse edges.
A logistic readout maps node and propagated graph features to an exposure score.

The comparison baseline is a logistic classifier using node attributes and basic
incoming/outgoing flow totals, without propagated disruption messages. Both models
are scored using leave-one-disruption-scenario-out evaluation. The report includes
accuracy, balanced accuracy, per-class recall, confusion matrix, and thresholded
recipient review rows.

## Limitations

- The synthetic label-generation rule uses a simple flow threshold and does not
  simulate inventory, substitution, route capacity, demand, perishability, or timing.
- The generated network is small and not representative of a real region or supplier
  ecosystem.
- The same synthetic mechanism creates labels and graph patterns; performance does
  not establish real-world prediction quality.
- Coordinates and community indices are fictional and must not be interpreted as
  actual community vulnerability.
- Scenario-held-out results are based on a small number of synthetic disruptions.
- A model score does not decide how food, funding, or services should be allocated.

## Responsible validation

Before any real application, obtain authorized data; validate relationship semantics,
edge dates, capacities, inventory and substitution behavior; compare against simple
graph baselines; evaluate across time and independently reviewed disruptions; and
audit error rates and impact by service area. Keep allocation choices with authorized
human planners and document uncertainty and missing network coverage.
