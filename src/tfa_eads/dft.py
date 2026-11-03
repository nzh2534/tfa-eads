"""Stage 3: DFT tier, with ORCA (default) or Quantum ESPRESSO.

dft-setup  : for every host, take the lowest xTB host conformers and the lowest
             valid xTB poses for each anion, plus the isolated anions, and write
             one job directory each (input files + run.sh + meta.json), plus
             jobs.txt and submit_array.sh (SLURM array) at the top level.
             Engine-specific inputs: see orca_engine.py / qe_engine.py.
dft-collect: parse all outputs and compute
             E_ads = E_complex - E_host - E_anion   (lowest of each, kcal/mol)
             dE_exch(X) = E_ads(TFA) - E_ads(X)
             and the same with free energies if available (ORCA + freq).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import orca_engine, qe_engine
from .geometry import HARTREE_TO_KCAL, bonds, read_xyz

log = logging.getLogger(__name__)
ENGINES = {"orca": orca_engine, "qe": qe_engine}


def _engine(dcfg):
    name = dcfg.get("engine", "orca")
    if name not in ENGINES:
        raise ValueError(f"dft.engine must be one of {list(ENGINES)}, got {name!r}")
    return name, ENGINES[name]


def _submit_script(dcfg, n_jobs, dft_dir):
    sl = dcfg["slurm"]
    mem_mb = int(dcfg["nprocs"] * dcfg["maxcore_mb"] * 1.25)
    throttle = f"%{sl['max_concurrent']}" if sl.get("max_concurrent") else ""
    if dcfg.get("engine", "orca") == "qe":   # pw.x: one MPI rank per core
        tasks = [f"#SBATCH --ntasks={dcfg['nprocs']}", "#SBATCH --cpus-per-task=1"]
        hint = "# module load quantum-espresso   # <- adapt; pw.x needs Environ for solvation"
        check = []
    else:
        tasks = ["#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={dcfg['nprocs']}"]
        hint = "# module load orca openmpi   # <- adapt to your cluster"
        check = [': "${ORCA_EXE:?set ORCA_EXE to the full path of the orca binary}"']
    lines = ["#!/bin/bash", "#SBATCH --job-name=tfa_dft", f"#SBATCH --array=1-{n_jobs}{throttle}",
             "#SBATCH --nodes=1", *tasks,
             f"#SBATCH --mem={mem_mb}M", f"#SBATCH --time={sl['time']}",
             f"#SBATCH --output={dft_dir}/logs/%A_%a.log"]
    if sl.get("partition"):
        lines.append(f"#SBATCH --partition={sl['partition']}")
    if sl.get("account"):
        lines.append(f"#SBATCH --account={sl['account']}")
    lines += ["", hint, *check,
              f'JOB=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {dft_dir}/jobs.txt)',
              'bash "$JOB/run.sh"', ""]
    return "\n".join(lines)



def run_dft_setup(cfg, ids=None):
    wd = Path(cfg["workdir"])
    dcfg = cfg["dft"]
    ename, eng = _engine(dcfg)
    if ename == "qe" and dcfg["freq"]:
        log.warning("dft.freq is not supported with Quantum ESPRESSO; only electronic E_ads")
    dft_dir = wd / "dft"
    prepared = pd.read_csv(wd / "molecules_prepared.csv", dtype={"id": str})
    prepared = prepared[prepared["status"] == "ok"]
    if ids:
        prepared = prepared[prepared["id"].isin([str(i) for i in ids])]
    from .workflow import anions

    # Each group of jobs shares one box size (QE only): every host's own
    # conformers + complexes are computed in the same cell, which cancels
    # most of the residual box-size dependence in E_ads.
    groups = []
    refs = {}
    for an in anions(cfg):
        ref = json.loads((wd / "refs" / an["name"] / "summary.json").read_text())
        refs[an["name"]] = ref
        syms, xyz = read_xyz(ref["xyz"])
        groups.append([(dft_dir / "refs" / an["name"], syms, xyz, ref["charge"], 1 + ref["uhf"],
                        {"role": "anion", "anion": an["name"]})])

    for _, r in prepared.iterrows():
        hid = r["id"]
        cs_path = wd / "hosts" / hid / "conformers" / "summary.json"
        if not cs_path.exists():
            continue
        cs = json.loads(cs_path.read_text())
        if not cs["conformers"]:
            continue
        group = []
        for k, conf in enumerate(cs["conformers"][: dcfg["n_host_conformers"]]):
            syms, xyz = read_xyz(conf["xyz"])
            group.append((dft_dir / "hosts" / hid / f"host_{k}", syms, xyz, cs["charge"],
                          1 + cs["uhf"], {"role": "host", "host": hid, "xtb_energy": conf["energy"]}))
        for an in anions(cfg):
            ps_path = wd / "hosts" / hid / "poses" / an["name"] / "summary.json"
            if not ps_path.exists():
                continue
            ps = json.loads(ps_path.read_text())
            ref = refs[an["name"]]
            n_guest = len(read_xyz(ref["xyz"])[0])
            for k, pose in enumerate(ps["best"][: dcfg["n_poses"]]):
                syms, xyz = read_xyz(pose["xyz"])
                group.append((dft_dir / "hosts" / hid / f"{an['name']}_{k}", syms, xyz,
                              cs["charge"] + ref["charge"], 1 + (cs["uhf"] + ref["uhf"]) % 2,
                              {"role": "complex", "host": hid, "anion": an["name"],
                               "n_host": len(syms) - n_guest, "xtb_energy": pose["energy"],
                               "start_site": pose["site_kind"]}))
        groups.append(group)

    jobdirs = []
    for group in groups:
        L = None
        if ename == "qe":
            L = qe_engine.box_length(max(qe_engine.extent(g[2]) for g in group), dcfg["qe"])
        for jd, syms, xyz, charge, mult, meta in group:
            eng.write_job(jd, syms, xyz, charge, mult, L, dcfg)
            meta = dict(meta, engine=ename, charge=charge, mult=mult, n_atoms=len(syms),
                        box_A=L)
            (jd / "meta.json").write_text(json.dumps(meta, indent=1))
            jobdirs.append(jd)

    (dft_dir / "logs").mkdir(parents=True, exist_ok=True)
    (dft_dir / "jobs.txt").write_text("\n".join(str(j) for j in jobdirs) + "\n")
    (dft_dir / "submit_array.sh").write_text(_submit_script(dcfg, len(jobdirs), dft_dir.resolve()))
    log.info("wrote %d %s jobs for %d hosts under %s", len(jobdirs), ename, len(groups) - len(refs), dft_dir)
    log.info("submit with: sbatch %s   (or run each run.sh yourself)", dft_dir / "submit_array.sh")
    return jobdirs


def parse_job(jd: Path, temperature: float) -> dict:
    meta = json.loads((jd / "meta.json").read_text())
    ename = meta.get("engine", "orca")
    if ename == "qe":
        res = qe_engine.parse(jd, meta)
    else:
        res = orca_engine.parse(jd, meta, temperature)
    rec = dict(meta, dir=str(jd), E_sp=None, G_corr=None)
    rec.update(res)
    # connectivity sanity check after the DFT optimization
    if rec["status"].startswith("ok") and meta["n_atoms"] > 1 and (jd / "opt.xyz").exists():
        s0, x0 = read_xyz(jd / "input.xyz")
        s1, x1 = read_xyz(jd / "opt.xyz")
        if bonds(s0, x0) != bonds(s1, x1):
            rec["status"] = "connectivity_changed"
    return rec


def run_dft_collect(cfg):
    wd = Path(cfg["workdir"])
    dft_dir = wd / "dft"
    T = cfg["dft"]["temperature"]
    recs = [parse_job(Path(p).parent, T) for p in sorted(dft_dir.rglob("meta.json"))]
    jobs = pd.DataFrame(recs)
    jobs.to_csv(wd / "dft_jobs_status.csv", index=False)
    log.info("job status: %s", jobs["status"].value_counts().to_dict())

    usable = jobs[jobs["status"].isin(["ok", "ok_imaginary_freq"])].copy()
    usable["G"] = usable["E_sp"] + usable["G_corr"].astype(float)   # NaN without freq

    from .workflow import anions
    guest = cfg["guest"]["name"]
    ref_E, ref_G = {}, {}
    for an in anions(cfg):
        r = usable[(usable["role"] == "anion") & (usable["anion"] == an["name"])]
        if len(r):
            ref_E[an["name"]] = r["E_sp"].min()
            if r["G"].notna().any():
                ref_G[an["name"]] = r["G"].min()

    prepared = pd.read_csv(wd / "molecules_prepared.csv", dtype={"id": str})
    xtb_path = wd / "results_xtb.csv"
    xtb = pd.read_csv(xtb_path, dtype={"id": str}) if xtb_path.exists() else None
    rows = []
    for _, p in prepared[prepared["status"] == "ok"].iterrows():
        hid = p["id"]
        h = usable[(usable["role"] == "host") & (usable["host"] == hid)]
        row = {"id": hid, "smiles": p["smiles"], "charge": p["charge"]}
        if not len(h):
            all_h = jobs[(jobs["role"] == "host") & (jobs["host"] == hid)]
            row["dft_status"] = "not_set_up" if not len(all_h) else "host_incomplete"
            rows.append(row)
            continue
        eh = h["E_sp"].min()
        gh = h["G"].min() if h["G"].notna().any() else None
        for an in anions(cfg):
            name = an["name"]
            c = usable[(usable["role"] == "complex") & (usable["host"] == hid) & (usable["anion"] == name)]
            if not len(c) or name not in ref_E:
                continue
            row[f"Eads_dft_{name}"] = round((c["E_sp"].min() - eh - ref_E[name]) * HARTREE_TO_KCAL, 2)
            row[f"n_poses_dft_{name}"] = len(c)
            if gh is not None and name in ref_G and c["G"].notna().any():
                row[f"Gads_dft_{name}"] = round((c["G"].min() - gh - ref_G[name]) * HARTREE_TO_KCAL, 2)
        for an in cfg.get("competitors") or []:
            cn = an["name"]
            for q in ("Eads", "Gads"):
                a, b = f"{q}_dft_{guest}", f"{q}_dft_{cn}"
                if a in row and b in row:
                    row[f"d{q[0]}exch_dft_vs_{cn}"] = round(row[a] - row[b], 2)
        row["dft_status"] = "ok" if f"Eads_dft_{guest}" in row else "complex_incomplete"
        rows.append(row)
    res = pd.DataFrame(rows)
    if xtb is not None:
        keep = ["id"] + [c for c in xtb.columns if c.startswith(("Eads_xtb", "dEexch_xtb", "best_start_site"))]
        res = res.merge(xtb[keep], on="id", how="left")
    meta_cols = [c for c in prepared.columns if c not in res.columns and c not in ("smiles_input",)]
    res = res.merge(prepared[["id"] + meta_cols], on="id", how="left")
    out = wd / "results_dft.csv"
    res.to_csv(out, index=False)
    log.info("DFT results -> %s", out)
    return res
