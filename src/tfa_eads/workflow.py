"""Stage 2: xTB tier.

  refs      : TFA and competitor anions, optimized once
  conformers: RDKit ETKDG -> force field -> xTB optimization of the lowest few
  docking   : anion poses on the lowest host conformer(s), xTB-optimized
  energies  : E_ads = E_complex - E_host - E_anion  (xTB level, kcal/mol)

Everything is written under <workdir>, and every xTB job is skipped if it has
already finished, so the command can simply be re-run after an interruption.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from . import docking
from .geometry import (HARTREE_TO_KCAL, bonds, embed_conformers, min_intermolecular_distance,
                       mol_coords, mol_symbols, write_xyz)
from .xtb import XtbJob, run_many

log = logging.getLogger(__name__)


def _uhf(mol, charge):
    n = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    return n % 2


def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def _load_json(path):
    return json.loads(Path(path).read_text())


def _too_close(symbols, coords, nh, scale):
    """Heavy-atom host-guest contact far inside the vdW sum (e.g. a carboxylate C
    2.4 A from a sulfonate O). Semi-covalent artifacts like this are a known xTB
    failure mode and inflate E_ads, so such poses are excluded."""
    from .geometry import VDW
    hi = [i for i in range(nh) if symbols[i] != "H"]
    gi = [i for i in range(nh, len(symbols)) if symbols[i] != "H"]
    if not hi or not gi:
        return False
    d = np.linalg.norm(coords[hi][:, None, :] - coords[gi][None, :, :], axis=-1)
    r = np.array([[VDW.get(symbols[i], 1.8) + VDW.get(symbols[j], 1.8) for j in gi] for i in hi])
    return bool((d < scale * r).any())


def anions(cfg):
    return [cfg["guest"]] + list(cfg.get("competitors") or [])


# ------------------------------------------------------------------ species
def _conformer_jobs(smiles, charge, base: Path, ccfg, n=None):
    """Embed and build xTB jobs for one species. Writes species.sdf (atom order)."""
    mol, confs = embed_conformers(smiles, ccfg, n=n)
    base.mkdir(parents=True, exist_ok=True)
    keep = confs[: ccfg["n_to_xtb"]]
    Chem.MolToMolFile(mol, str(base / "species.sdf"), confId=keep[0][0])
    syms = mol_symbols(mol)
    uhf = _uhf(mol, charge)
    jobs = [XtbJob(base / f"conf_{k:03d}", syms, mol_coords(mol, cid), charge, uhf,
                   tag=f"ff_energy={e:.3f}")
            for k, (cid, e) in enumerate(keep)]
    return mol, jobs


def _summarize_conformers(mol, jobs, results, n_keep):
    syms = mol_symbols(mol)
    rows = []
    for job, res in zip(jobs, results):
        if not res.ok:
            continue
        if bonds(syms, job.coords) != bonds(syms, res.coords):
            log.warning("connectivity changed during xTB opt in %s; discarded", job.workdir)
            continue
        rows.append({"dir": str(job.workdir), "energy": res.energy,
                     "xyz": str(Path(job.workdir) / ("xtbopt.xyz" if len(syms) > 1 else "input.xyz"))})
    rows.sort(key=lambda r: r["energy"])
    distinct = []
    for r in rows:   # drop duplicates within 0.1 kcal/mol
        if all(abs(r["energy"] - d["energy"]) * HARTREE_TO_KCAL > 0.1 for d in distinct):
            distinct.append(r)
    return distinct[:n_keep]


# ------------------------------------------------------------------ stage A
def run_refs(cfg):
    wd = Path(cfg["workdir"]) / "refs"
    all_jobs, meta = [], []
    for an in anions(cfg):
        mol0 = Chem.MolFromSmiles(an["smiles"])
        charge = Chem.GetFormalCharge(mol0)
        mol, jobs = _conformer_jobs(an["smiles"], charge, wd / an["name"], cfg["conformers"], n=10)
        meta.append((an, mol, charge, len(all_jobs), len(jobs)))
        all_jobs += jobs
    results = run_many(all_jobs, cfg["xtb"], "refs")
    out = {}
    for an, mol, charge, start, n in meta:
        best = _summarize_conformers(mol, all_jobs[start:start + n], results[start:start + n], 1)
        if not best:
            raise RuntimeError(f"reference {an['name']} failed in xTB")
        rec = {"name": an["name"], "smiles": an["smiles"], "charge": charge,
               "uhf": _uhf(mol, charge), "energy": best[0]["energy"], "xyz": best[0]["xyz"],
               "sdf": str(wd / an["name"] / "species.sdf")}
        _save_json(wd / an["name"] / "summary.json", rec)
        out[an["name"]] = rec
    log.info("references done: %s", ", ".join(f"{k}={v['energy']:.6f} Eh" for k, v in out.items()))
    return out


# ------------------------------------------------------------------ stage B
def run_host_conformers(cfg, prepared: pd.DataFrame):
    hosts_dir = Path(cfg["workdir"]) / "hosts"
    all_jobs, meta = [], []
    for _, r in prepared.iterrows():
        base = hosts_dir / r["id"] / "conformers"
        summ = base / "summary.json"
        if summ.exists():
            continue
        try:
            mol, jobs = _conformer_jobs(r["smiles"], int(r["charge"]), base, cfg["conformers"])
        except Exception as exc:  # noqa: BLE001
            log.error("host %s: embedding failed (%s)", r["id"], exc)
            _save_json(summ, {"id": r["id"], "status": f"embed_failed: {exc}", "conformers": []})
            continue
        meta.append((r, mol, len(all_jobs), len(jobs)))
        all_jobs += jobs
    log.info("host conformers: %d xTB jobs for %d hosts", len(all_jobs), len(meta))
    results = run_many(all_jobs, cfg["xtb"], "host conformers")
    for r, mol, start, n in meta:
        best = _summarize_conformers(mol, all_jobs[start:start + n], results[start:start + n],
                                     cfg["conformers"]["n_keep"])
        _save_json(hosts_dir / r["id"] / "conformers" / "summary.json",
                   {"id": r["id"], "status": "ok" if best else "xtb_failed",
                    "charge": int(r["charge"]), "uhf": _uhf(mol, int(r["charge"])),
                    "sdf": str(hosts_dir / r["id"] / "conformers" / "species.sdf"),
                    "conformers": best})


# ------------------------------------------------------------------ stage C
def run_docking(cfg, prepared: pd.DataFrame, refs: dict):
    from .geometry import read_xyz
    hosts_dir = Path(cfg["workdir"]) / "hosts"
    dcfg = cfg["docking"]
    guests = {}
    for an in anions(cfg):
        ref = refs[an["name"]]
        gmol = Chem.MolFromMolFile(ref["sdf"], removeHs=False)
        gsyms, gxyz = read_xyz(ref["xyz"])
        guests[an["name"]] = (gmol, gsyms, gxyz, ref)

    all_jobs, meta = [], []
    for _, r in prepared.iterrows():
        hid = r["id"]
        cs_path = hosts_dir / hid / "conformers" / "summary.json"
        if not cs_path.exists():
            continue
        cs = _load_json(cs_path)
        if not cs["conformers"]:
            continue
        hmol = Chem.MolFromMolFile(cs["sdf"], removeHs=False)
        for name, (gmol, gsyms, gxyz, ref) in guests.items():
            pdir = hosts_dir / hid / "poses" / name
            if (pdir / "summary.json").exists():
                continue
            jobs, infos = [], []
            for ci, conf in enumerate(cs["conformers"][: dcfg["host_conformers"]]):
                hsyms, hxyz = read_xyz(conf["xyz"])
                for pj, (gpos, info) in enumerate(docking.generate_poses(
                        hmol, hsyms, hxyz, gmol, gsyms, gxyz, dcfg)):
                    syms = hsyms + gsyms
                    xyz = np.vstack([hxyz, gpos])
                    charge = cs["charge"] + ref["charge"]
                    uhf = (cs["uhf"] + ref["uhf"]) % 2
                    jobs.append(XtbJob(pdir / f"c{ci}_p{pj:02d}", syms, xyz, charge, uhf,
                                       tag=f"{info['site_kind']} site atom {info['site_atom']}"))
                    infos.append(dict(info, host_conf=ci, n_host=len(hsyms),
                                      host_bonds=sorted(bonds(hsyms, hxyz)),
                                      guest_bonds=sorted(bonds(gsyms, gxyz))))
            if not jobs:
                log.warning("host %s / %s: no clash-free poses generated", hid, name)
            meta.append((hid, name, pdir, len(all_jobs), jobs, infos))
            all_jobs += jobs
    log.info("docking: %d xTB pose optimizations", len(all_jobs))
    results = run_many(all_jobs, cfg["xtb"], "poses")

    for hid, name, pdir, start, jobs, infos in meta:
        rows = []
        for job, info, res in zip(jobs, infos, results[start:start + len(jobs)]):
            row = {"dir": str(job.workdir), "site_kind": info["site_kind"],
                   "site_element": info["site_element"], "site_atom": info["site_atom"],
                   "host_conf": info["host_conf"], "ok": res.ok, "energy": res.energy,
                   "status": "ok" if res.ok else res.error}
            if res.ok:
                nh = info["n_host"]
                expect = set(map(tuple, info["host_bonds"])) | {
                    (i + nh, j + nh) for i, j in info["guest_bonds"]}
                if bonds(res.symbols, res.coords) != expect:
                    row["status"] = "reacted_or_proton_transfer"
                elif min_intermolecular_distance(res.coords[:nh], res.coords[nh:]) > dcfg["dissociated_cutoff"]:
                    row["status"] = "dissociated"
                elif _too_close(res.symbols, res.coords, nh, dcfg["close_contact_scale"]):
                    row["status"] = "close_contact"
                row["xyz"] = str(Path(job.workdir) / "xtbopt.xyz")
            rows.append(row)
        valid = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: r["energy"])
        _save_json(pdir / "summary.json", {"host": hid, "anion": name, "n_poses": len(rows),
                                           "n_valid": len(valid), "poses": rows,
                                           "best": valid[: dcfg["n_keep"]]})


# ------------------------------------------------------------------ energies
def collect_xtb(cfg, prepared: pd.DataFrame, refs: dict) -> pd.DataFrame:
    hosts_dir = Path(cfg["workdir"]) / "hosts"
    guest = cfg["guest"]["name"]
    out = []
    for _, r in prepared.iterrows():
        hid = r["id"]
        row = {"id": hid, "smiles": r["smiles"], "charge": r["charge"]}
        for c in prepared.columns:
            if c not in row and c not in ("smiles_input",):
                row[c] = r[c]
        cs_path = hosts_dir / hid / "conformers" / "summary.json"
        notes = []
        if not cs_path.exists():
            row["xtb_status"] = "not_run"
            out.append(row)
            continue
        if not _load_json(cs_path)["conformers"]:
            row["xtb_status"] = "host_failed"
            out.append(row)
            continue
        e_host = _load_json(cs_path)["conformers"][0]["energy"]
        row["E_host_xtb_Eh"] = e_host
        for an in anions(cfg):
            name = an["name"]
            ps_path = hosts_dir / hid / "poses" / name / "summary.json"
            if not ps_path.exists():
                notes.append(f"{name}:not_docked")
                continue
            ps = _load_json(ps_path)
            for bad in ("reacted_or_proton_transfer", "close_contact"):
                nb = sum(p["status"] == bad for p in ps["poses"])
                if nb:
                    notes.append(f"{name}:{nb}_poses_{bad}")
            row[f"n_valid_poses_{name}"] = ps["n_valid"]
            if not ps["best"]:
                notes.append(f"{name}:no_valid_pose")
                continue
            b = ps["best"][0]
            e_ads = (b["energy"] - e_host - refs[name]["energy"]) * HARTREE_TO_KCAL
            row[f"Eads_xtb_{name}"] = round(e_ads, 2)
            row[f"best_start_site_{name}"] = f"{b['site_kind']}:{b['site_element']}{b['site_atom']}"
        for an in cfg.get("competitors") or []:
            c = an["name"]
            if f"Eads_xtb_{guest}" in row and f"Eads_xtb_{c}" in row:
                row[f"dEexch_xtb_vs_{c}"] = round(row[f"Eads_xtb_{guest}"] - row[f"Eads_xtb_{c}"], 2)
        row["xtb_status"] = "ok" if f"Eads_xtb_{guest}" in row else "incomplete"
        row["xtb_notes"] = ";".join(notes)
        out.append(row)
    df = pd.DataFrame(out)
    path = Path(cfg["workdir"]) / "results_xtb.csv"
    df.to_csv(path, index=False)
    log.info("xTB results -> %s", path)
    return df


def run_xtb_tier(cfg, ids=None, limit=None):
    prepared = pd.read_csv(Path(cfg["workdir"]) / "molecules_prepared.csv", dtype={"id": str})
    prepared = prepared[prepared["status"] == "ok"]
    todo = prepared
    if ids:
        todo = todo[todo["id"].isin([str(i) for i in ids])]
    if limit:
        todo = todo.head(limit)
    log.info("xTB tier on %d hosts", len(todo))
    refs = run_refs(cfg)
    run_host_conformers(cfg, todo)
    run_docking(cfg, todo, refs)
    return collect_xtb(cfg, prepared, refs)   # table always covers every host
