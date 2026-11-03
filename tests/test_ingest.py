import pandas as pd
import pytest

from tfa_eads.ingest import prepare_one, run_prepare


@pytest.mark.parametrize("smiles,expected_charge,change", [
    ("CS(=O)(=O)O", -1, "sulfonic"),                # strong acid -> anion
    ("CC(=O)O", -1, "carboxylic"),
    ("CCN", 1, "aliphatic amine"),
    ("NC(N)=N", 1, "guanidine"),
    ("Nc1ccccc1", 0, ""),                          # aniline stays neutral
    ("CC(=O)NN", 0, ""),                           # hydrazide stays neutral
    ("C[N+](C)(C)C", 1, ""),                       # already charged
    ("c1ccncc1", 0, ""),                           # pyridine stays neutral at pH 7
])
def test_rules_protonation(smiles, expected_charge, change):
    rec = prepare_one(smiles, "rules", 7.0)
    assert rec["status"] == "ok"
    assert rec["charge"] == expected_charge
    assert change in rec["protonation_changes"]


def test_as_given_keeps_charge():
    rec = prepare_one("CC(=O)O", "as_given", 7.0)
    assert rec["charge"] == 0


def test_override_wins():
    rec = prepare_one("CC(=O)O", "rules", 7.0, override="CC(=O)O")
    assert rec["charge"] == 0 and rec["protonation_changes"] == "manual override"


def test_flags():
    assert prepare_one("not_a_smiles", "rules", 7.0)["status"] == "invalid_smiles"
    assert prepare_one("*CC*", "rules", 7.0)["status"] == "skip"
    rec = prepare_one("CC(=O)[O-].[Na+]", "as_given", 7.0)
    assert "multi_fragment_kept_largest" in rec["warnings"] and rec["charge"] == -1
    rec = prepare_one("OC(=O)C(F)(F)C(F)(F)F", "rules", 7.0)
    assert "contains_perfluoroalkyl_chain" in rec["warnings"]
    rec = prepare_one("CC", "as_given", 7.0, given_charge=1)
    assert any(w.startswith("charge_column_mismatch") for w in rec["warnings"])


def test_nitro_is_not_a_cationic_site():
    assert prepare_one("c1ccccc1[N+](=O)[O-]", "rules", 7.0)["n_cationic_sites"] == 0
    assert prepare_one("C[N+](C)(C)C", "rules", 7.0)["n_cationic_sites"] == 1


def test_run_prepare_example(cfg, tmp_path):
    cfg["input_csv"] = str(__import__("pathlib").Path(__file__).parents[1] / "examples" / "example_hosts.csv")
    out = run_prepare(cfg)
    assert (out["status"] == "ok").all()
    assert dict(zip(out["id"], out["charge"])) == {"6380": 1, "3520": 1, "1049": 0, "176": -1, "241": 0}
    assert (tmp_path / "work" / "molecules_prepared.csv").exists()


def test_duplicate_ids_rejected(cfg, tmp_path):
    p = tmp_path / "dup.csv"
    pd.DataFrame({"cid": [1, 1], "smiles": ["C", "CC"]}).to_csv(p, index=False)
    cfg["input_csv"] = str(p)
    with pytest.raises(ValueError):
        run_prepare(cfg)
