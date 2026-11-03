"""End-to-end prepare + xtb tier on one tiny host. Needs `xtb` on PATH."""
import pandas as pd
import pytest

from tfa_eads.ingest import run_prepare
from tfa_eads.workflow import run_xtb_tier


@pytest.mark.xtb
def test_xtb_tier_tetramethylammonium(cfg, tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"cid": [6380], "smiles": ["C[N+](C)(C)C"]}).to_csv(p, index=False)
    cfg["input_csv"] = str(p)
    cfg["conformers"].update(n_to_xtb=2, rdkit_n_max=10)
    cfg["docking"].update(max_poses=4)
    cfg["xtb"]["parallel_jobs"] = 2
    run_prepare(cfg)
    res = run_xtb_tier(cfg)
    row = res.iloc[0]
    assert row["xtb_status"] == "ok"
    # a cation should bind TFA- favorably even at the xTB level
    assert row["Eads_xtb_TFA"] < 0
    assert pd.notna(row["dEexch_xtb_vs_chloride"])
