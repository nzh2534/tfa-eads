"""Stage 1: read SMILES, validate, assign protonation state, flag problems.

Writes <workdir>/molecules_prepared.csv, which every later stage reads.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based protonation at ~pH 7.
# Deliberately conservative: only groups whose protonation state at pH 7 is
# unambiguous are changed. Borderline groups (imidazoles, morpholines,
# anilines, phenols, imides) are left as drawn. Review the
# `protonation_changes` column with your chemist; override any row through
# the `smiles_override` column of the input CSV.
# ---------------------------------------------------------------------------
_DEPROTONATE = [
    # (label, SMARTS, index of the atom that loses H+, one-per-key-atom index or None)
    ("sulfonic/sulfate", "[OX2H1][SX4](=O)(=O)", 0, None),
    ("phosphate/phosphonate", "[OX2H1][PX4]=O", 0, 1),     # first acidic OH per P only
    ("carboxylic acid", "[OX2H1][CX3](=[OX1])[#6]", 0, None),
    ("sulfonylurea/acylsulfonamide", "[NX3H1]([SX4](=O)=O)[CX3]=O", 0, None),
]
_PROTONATE = [
    # aliphatic amine: N bonded only to sp3 carbons (and H)
    ("aliphatic amine", "[NX3;+0;!$(N~[!#6;!#1]);!$(N-[#6;!X4])]", 0, None),
    # guanidine / guanylhydrazone: protonate the imine N, one per central C
    ("guanidine",
     "[NX2;+0;!$(N[CX3]=O);!$(N[SX4](=O)=O);!$(N-O)]=[CX3]"
     "([NX3;!$(N[CX3]=O);!$(N[SX4](=O)=O)])[NX3;!$(N[CX3]=O);!$(N[SX4](=O)=O)]", 0, 1),
    # acyclic amidine
    ("amidine",
     "[NX2;+0;!$(N~[!#6;!#1])]=[CX3;!R]([#6])[NX3;!$(N~[!#6;!#1]);!$(N[CX3]=O)]", 0, 1),
]
_PATTERNS = {label: Chem.MolFromSmarts(s) for label, s, _, _ in _DEPROTONATE + _PROTONATE}


def _apply(mol: Chem.Mol, rules, delta: int) -> tuple[Chem.Mol, list[str]]:
    rw = Chem.RWMol(mol)
    changes, done_atoms, done_keys = [], set(), set()
    for label, _, idx, key_idx in rules:
        for match in rw.GetSubstructMatches(_PATTERNS[label]):
            a = match[idx]
            if a in done_atoms:
                continue
            if key_idx is not None:
                if match[key_idx] in done_keys:
                    continue
                done_keys.add(match[key_idx])
            atom = rw.GetAtomWithIdx(a)
            nh = atom.GetTotalNumHs()
            if delta < 0 and nh == 0:
                continue
            atom.SetFormalCharge(atom.GetFormalCharge() + (1 if delta > 0 else -1))
            atom.SetNumExplicitHs(nh + (1 if delta > 0 else -1))
            atom.SetNoImplicit(True)
            done_atoms.add(a)
            changes.append(label)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m, changes


def protonate_rules(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    mol, c1 = _apply(mol, _DEPROTONATE, -1)
    mol, c2 = _apply(mol, _PROTONATE, +1)
    from collections import Counter
    out = []
    for sign, changes in (("-H+", c1), ("+H+", c2)):
        for label, n in Counter(changes).items():
            out.append(f"{sign} {label}" + (f" x{n}" if n > 1 else ""))
    return mol, out


def protonate_dimorphite(smiles: str, ph: float) -> str:
    from dimorphite_dl import protonate_smiles
    out = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.0, max_variants=1)
    if not out:
        raise ValueError("dimorphite returned no state")
    return out[0]


# ---------------------------------------------------------------------------
_PERFLUORO = Chem.MolFromSmarts("[CX4](F)(F)[CX4](F)(F)")


def cationic_sites(mol: Chem.Mol) -> int:
    """Positive atoms that are real cationic sites (excludes nitro / N-oxide N+)."""
    n = 0
    for at in mol.GetAtoms():
        if at.GetFormalCharge() > 0 and not any(nb.GetFormalCharge() < 0 for nb in at.GetNeighbors()):
            n += 1
    return n


def prepare_one(smiles: str, mode: str, ph: float, given_charge=None, override=None) -> dict:
    rec = {"smiles_input": smiles, "warnings": [], "protonation_changes": ""}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec.update(status="invalid_smiles")
        return rec

    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        rec["warnings"].append("dummy_atoms_possible_periodic_or_polymer")
        rec.update(status="skip")
        return rec

    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        rec["warnings"].append("multi_fragment_kept_largest")

    fc_in = Chem.GetFormalCharge(mol)
    if given_charge is not None and pd.notna(given_charge) and int(given_charge) != fc_in:
        rec["warnings"].append(f"charge_column_mismatch({int(given_charge)}vs{fc_in})")

    if override is not None and isinstance(override, str) and override.strip():
        m2 = Chem.MolFromSmiles(override.strip())
        if m2 is None:
            rec.update(status="invalid_override")
            return rec
        mol = m2
        rec["protonation_changes"] = "manual override"
    elif mode == "rules":
        mol, changes = protonate_rules(mol)
        rec["protonation_changes"] = "; ".join(changes)
    elif mode == "dimorphite":
        try:
            m2 = Chem.MolFromSmiles(protonate_dimorphite(Chem.MolToSmiles(mol), ph))
            if m2 is None:
                raise ValueError("unparseable dimorphite output")
            if Chem.GetFormalCharge(m2) != fc_in:
                rec["protonation_changes"] = f"dimorphite {fc_in:+d}->{Chem.GetFormalCharge(m2):+d}"
            mol = m2
        except Exception as exc:  # noqa: BLE001
            rec["warnings"].append(f"dimorphite_failed({exc})")
    elif mode != "as_given":
        raise ValueError(f"unknown protonation mode {mode}")

    charge = Chem.GetFormalCharge(mol)
    mol_h = Chem.AddHs(mol)
    n_elec = sum(a.GetAtomicNum() for a in mol_h.GetAtoms()) - charge
    radicals = sum(a.GetNumRadicalElectrons() for a in mol_h.GetAtoms())
    mult = 2 if (n_elec % 2 or radicals == 1) else 1
    if mult != 1 or radicals > 1:
        rec["warnings"].append("open_shell")

    if abs(charge) >= 3:
        rec["warnings"].append("highly_charged")
    if mol.HasSubstructMatch(_PERFLUORO):
        rec["warnings"].append("contains_perfluoroalkyl_chain")

    rec.update(
        status="ok",
        smiles=Chem.MolToSmiles(mol),
        charge=charge,
        charge_input=fc_in,
        mult=mult,
        n_heavy=mol.GetNumHeavyAtoms(),
        n_atoms=mol_h.GetNumAtoms(),
        n_rot=rdMolDescriptors.CalcNumRotatableBonds(mol),
        n_cationic_sites=cationic_sites(mol),
        n_hbond_donors=rdMolDescriptors.CalcNumHBD(mol),
        mw=round(Descriptors.MolWt(mol), 2),
        elements="".join(sorted({a.GetSymbol() for a in mol_h.GetAtoms()})),
    )
    return rec


def run_prepare(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["input_csv"])
    idc, smc = cfg["id_column"], cfg["smiles_column"]
    chc, ovc = cfg.get("charge_column"), cfg.get("override_column")
    lim = cfg["limits"]
    rows = []
    for _, r in df.iterrows():
        rec = prepare_one(
            r[smc], cfg["protonation"], cfg["ph"],
            given_charge=r[chc] if chc in df.columns else None,
            override=r[ovc] if ovc in df.columns else None,
        )
        rec["id"] = str(r[idc])
        for extra in df.columns:  # carry metadata (e.g. cluster) through
            if extra not in (idc, smc, chc, ovc):
                rec[extra] = r[extra]
        if rec.get("status") == "ok":
            if lim["warn_heavy_atoms"] and rec["n_heavy"] > lim["warn_heavy_atoms"]:
                rec["warnings"].append("large_expensive_for_dft")
            if lim["skip_heavy_atoms"] and rec["n_heavy"] > lim["skip_heavy_atoms"]:
                rec["status"] = "skip"
                rec["warnings"].append("above_skip_heavy_atoms")
        rec["warnings"] = ";".join(rec["warnings"])
        rows.append(rec)

    out = pd.DataFrame(rows)
    front = ["id", "status", "smiles", "charge", "charge_input", "mult", "protonation_changes", "warnings"]
    out = out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]
    if out["id"].duplicated().any():
        raise ValueError("duplicate ids in input; ids must be unique")
    Path(cfg["workdir"]).mkdir(parents=True, exist_ok=True)
    path = Path(cfg["workdir"]) / "molecules_prepared.csv"
    out.to_csv(path, index=False)
    n_ok = (out["status"] == "ok").sum()
    n_chg = (out["protonation_changes"].fillna("") != "").sum()
    log.info("prepared %d molecules (%d ok, %d with protonation changes) -> %s", len(out), n_ok, n_chg, path)
    return out
