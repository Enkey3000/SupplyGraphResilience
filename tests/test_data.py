from supplygraph.data import generate_synthetic_network, load_network, write_network


def test_synthetic_network_is_deterministic():
    assert generate_synthetic_network(seed=5) == generate_synthetic_network(seed=5)


def test_generated_network_round_trips(tmp_path):
    original = generate_synthetic_network()
    write_network(original, tmp_path)
    loaded = load_network(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        tmp_path / "disruptions.csv",
    )
    assert loaded == original


def test_writer_creates_all_three_tables(tmp_path):
    write_network(generate_synthetic_network(), tmp_path)
    assert (tmp_path / "nodes.csv").is_file()
    assert (tmp_path / "edges.csv").is_file()
    assert (tmp_path / "disruptions.csv").is_file()


def test_loader_rejects_unknown_edge_endpoint(tmp_path):
    network = generate_synthetic_network()
    write_network(network, tmp_path)
    edge_file = tmp_path / "edges.csv"
    edge_file.write_text(
        "source_id,target_id,weekly_units,transit_days\nunknown,warehouse-01,10,1\n",
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="unknown node"):
        load_network(tmp_path / "nodes.csv", edge_file, tmp_path / "disruptions.csv")
