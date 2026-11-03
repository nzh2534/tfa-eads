# tfa_eads: SMILES → adsorption energies for trifluoroacetate

A restartable pipeline that takes a CSV of SMILES and computes

    E_ads      = E(host·TFA⁻) − E(host) − E(TFA⁻)
    ΔE_exch(X) = E_ads(TFA⁻) − E_ads(X⁻)     (X = chloride by default)

first at the GFN2-xTB level (cheap, minutes per molecule) and then at the DFT
level with ORCA (expensive, run on a cluster). Negative E_ads means binding is
favorable; negative ΔE_exch means the site prefers TFA⁻ over the competitor.
All energies are in kcal/mol and include implicit water.

## Install

```bash
pip install -r requirements.txt
# xtb: download a release binary and put it on PATH (or set xtb.executable)
#   https://github.com/grimme-lab/xtb/releases     or  conda install -c conda-forge xtb
# DFT, only on the machine that runs those jobs, ONE of:
#   ORCA (default)          dft.engine: orca
#   Quantum ESPRESSO + Environ  dft.engine: qe   (see "Using Quantum ESPRESSO")
```

## Run

```bash
python -m tfa_eads prepare     -c config.yaml   # validate + protonate     -> work/molecules_prepared.csv
python -m tfa_eads xtb         -c config.yaml   # conformers + docking + xTB -> work/results_xtb.csv
python -m tfa_eads dft-setup   -c config.yaml   # write ORCA jobs           -> work/dft/
sbatch work/dft/submit_array.sh                 # (export ORCA_EXE=/full/path/to/orca first)
python -m tfa_eads dft-collect -c config.yaml   # parse                     -> work/results_dft.csv
```

Add `--ids 413436 1296` or `--limit 5` to try a few molecules first. Every
step skips work that already finished, so after a crash or timeout you just
re-run the same command.

## What each stage does

**1. prepare.** Parses each SMILES with RDKit, keeps the largest fragment,
checks the `charge` column against the SMILES, and assigns the charge state
in water at pH 7. This matters for an anion like TFA⁻: a protonated amine is a
cationic binding site, and a sulfonic acid drawn neutral is really an anion
that repels TFA⁻. Every change is listed in `protonation_changes`. To fix one
by hand, add a `smiles_override` column to the input CSV and fill in that row.

**2. xtb.**
- *Conformers:* RDKit ETKDG generates 20–300 conformers (scaled by flexibility),
  MMFF relaxes them, and the lowest 10 are re-optimized with GFN2-xTB in
  implicit water (ALPB). The 3 lowest distinct ones are kept.
- *Docking:* starting poses for the anion are placed at cationic atoms, at
  polar N–H/O–H hydrogens, and at a few spread-out surface atoms, with the
  anion's charged end facing the site. Each pose is relaxed with xTB.
  Poses are excluded, and counted in `xtb_notes`, when a bond formed or broke
  (usually a proton jumping between host and TFA⁻), when the anion drifted
  away, or when a heavy atom ends up far inside van der Waals contact
  (`close_contact`). The last one catches a known xTB artifact seen in testing:
  a sulfonyl/sulfonate oxygen collapsing to ~2.4 Å from TFA⁻'s carboxylate
  carbon, which inflates E_ads by several kcal/mol. Sulfonamides are common in
  this seed set, so this filter matters.
- *Energies:* lowest valid complex − lowest host conformer − isolated anion.

**3. dft-setup / dft-collect** (ORCA defaults described here; for Quantum
ESPRESSO see below). For each host: the 3 lowest xTB host
conformers, the 3 lowest valid xTB poses per anion, plus each isolated anion,
become ORCA jobs. Each job is a geometry optimization (r2SCAN-3c, CPCM water)
followed by a single point (ωB97M-V/def2-TZVPD, SMD water). Diffuse basis
functions (the "D") matter for anions. `dft-collect` takes the minimum over
conformers/poses, writes `results_dft.csv` (with the xTB numbers alongside for
comparison) and `dft_jobs_status.csv` (one row per job, for spotting
failures). Set `dft.freq: true` to also get free energies (`Gads_*`).

## Using Quantum ESPRESSO instead of ORCA

Set `dft.engine: qe` and fill in the `dft.qe` block of `config.yaml`
(`pseudo_dir` at minimum). Everything else, including `dft-collect`, is the same.
Each job is one `pw.x` geometry optimization (`relax.in` → `relax.out`), and
E_ads comes from its final energy.

What changes physically, and why the defaults are what they are:

- **Functional:** PBE-D3(BJ) instead of r2SCAN-3c + ωB97M-V. Plane-wave hybrids
  in large vacuum boxes are very expensive, so there is no hybrid single point.
  PBE is less accurate for anions (it over-delocalizes the extra electron), so
  expect larger systematic errors than with the ORCA defaults. On the plus
  side, plane waves have no basis set superposition error.
- **Solvation needs Environ.** Core QE has no implicit solvent. Build `pw.x`
  with the Environ plugin (environ.readthedocs.io); jobs then run as
  `pw.x --environ` with an `environ.in` file (SCCS water). Without Environ
  (`solvation: none`) every species is in vacuum, which is unphysical for the
  charged hosts here (up to −24 and +4), so use that only for testing.
  The `environ.in` namelists were written for Environ 1.x/2.x; confirm them
  against the version on your cluster.
- **Charged cells:** each molecule sits in a cubic vacuum box, and charged
  cells need a correction. With Environ that is its 0D parabolic correction;
  without it, Martyna-Tuckerman (`assume_isolated='mt'`), which requires boxes
  at least twice the molecule's size (37 Å for some test hosts). A host's
  conformers and complexes all share one box size, so residual box effects
  largely cancel in E_ads. Still, check convergence once: rerun one host with
  `vacuum` raised by 3–4 Å and confirm E_ads moves by well under 1 kcal/mol.
- **Cutoffs:** all jobs must use the same `ecutwfc`/`ecutrho`. Take the largest
  SSSP-recommended value over the elements present (H C N O F S Cl Br I P).
- **No free energies:** `dft.freq` is ignored (Γ-point phonons for hundreds of
  molecules are impractical in QE).
- **Cost:** expect each job to be noticeably more expensive than the ORCA
  r2SCAN-3c optimization, mostly because of the vacuum. The cyclodextrins
  (~300 atoms, 45+ Å boxes) are the expensive tail; `limits.skip_heavy_atoms`
  can leave them out.

## Outputs worth looking at

| file | contents |
|---|---|
| `work/molecules_prepared.csv` | charge states, flags, size per molecule |
| `work/results_xtb.csv` | `Eads_xtb_TFA`, `Eads_xtb_chloride`, `dEexch_xtb_vs_chloride`, pose counts |
| `work/results_dft.csv` | same at DFT level, plus `Gads_*` if freq was on |
| `work/hosts/<id>/poses/<anion>/summary.json` | every pose, its energy and status |
| `work/hosts/<id>/poses/<anion>/c0_pXX/xtbopt.xyz` | complex geometries (open in Avogadro / VMD / ASE) |

The columns from `results_*.csv` are what you feed into the Gaussian process,
either as features or, for multi-fidelity BO, xTB as the low-fidelity and DFT
as the higher-fidelity signal.

## Notes on this seed set

- All 167 entries are discrete molecules; there are no periodic structures.
- The `charge` column agrees with the SMILES for every row.
- With `protonation: rules`, 22 molecules change charge at pH 7. The largest
  shifts are the three sulfated cyclodextrins (−9, −14, −24): as drawn they are
  neutral sulfate acids, but in water they are polyanions. Expect them to bind
  TFA⁻ weakly or not at all, and treat their numbers with extra caution
  (implicit solvent is least reliable for highly charged species).
- 9 molecules have more than 100 heavy atoms (cyclodextrins, cyclosporin-like
  peptides). They run fine with xTB but are expensive in DFT; set
  `limits.skip_heavy_atoms` if you want to leave them out of the DFT tier.
- Three hosts contain perfluoroalkyl chains (they are themselves PFAS-like).

## Caveats

- Treat absolute xTB E_ads values as rough. The exchange energy (ΔE_exch)
  benefits from error cancellation and is the more trustworthy xTB signal;
  DFT is there to correct both.
- Many clusters cap SLURM array size (often 1001). The full seed set produces
  roughly 1,500 ORCA jobs, so split `jobs.txt` or submit with `--ids` batches
  if `sbatch` rejects the array.
- Docking is "many starts + local optimization", not a global search. More
  starting poses (`docking.max_poses`) or more host conformers
  (`docking.host_conformers`) give more reliable minima at higher cost.
- Implicit solvation captures bulk water but not specific water molecules
  competing for the binding site. Expect trends, not lab-accurate numbers;
  check how well `dEexch` tracks your first lab results before trusting it.
- The docked complexes use the host's xTB geometry; the host can relax on
  binding, which is included because each complex is fully re-optimized.
- Highly charged hosts (|charge| ≥ 3) and open-shell molecules are flagged in
  `warnings`; review those by hand.
