"""Thin, restartable wrapper around the xtb binary (GFN2-xTB + ALPB implicit water)."""
from __future__ import annotations

import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import read_xyz, write_xyz

log = logging.getLogger(__name__)
_E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")


@dataclass
class XtbJob:
    workdir: Path
    symbols: list
    coords: object
    charge: int
    uhf: int = 0
    opt: bool = True
    tag: str = ""


@dataclass
class XtbResult:
    workdir: Path
    ok: bool
    energy: float | None      # Hartree
    symbols: list | None = None
    coords: object = None
    error: str = ""


def _parse(workdir: Path, opt: bool) -> XtbResult:
    out = workdir / "xtb.out"
    if not out.exists():
        return XtbResult(workdir, False, None, error="no output")
    text = out.read_text(errors="replace")
    if "normal termination of xtb" not in text:
        return XtbResult(workdir, False, None, error="abnormal termination")
    if opt and "GEOMETRY OPTIMIZATION CONVERGED" not in text:
        return XtbResult(workdir, False, None, error="optimization not converged")
    m = _E_RE.findall(text)
    if not m:
        return XtbResult(workdir, False, None, error="energy not found")
    geo = workdir / ("xtbopt.xyz" if opt else "input.xyz")
    syms, xyz = read_xyz(geo)
    return XtbResult(workdir, True, float(m[-1]), syms, xyz)


def run_job(job: XtbJob, xcfg: dict) -> XtbResult:
    wd = Path(job.workdir)
    done = _parse(wd, job.opt and len(job.symbols) > 1)
    if done.ok and (wd / "input.xyz").exists():
        old_syms, old_xyz = read_xyz(wd / "input.xyz")
        if old_syms == list(job.symbols) and np.allclose(old_xyz, job.coords, atol=1e-4):
            return done  # restart: this exact job already finished
        for f in wd.iterdir():  # stale job from a different setup: start over
            if f.is_file():
                f.unlink()
    wd.mkdir(parents=True, exist_ok=True)
    write_xyz(wd / "input.xyz", job.symbols, job.coords, job.tag)
    single_atom = len(job.symbols) == 1
    cmd = [xcfg["executable"], "input.xyz", "--gfn", str(xcfg["gfn"]),
           "--chrg", str(job.charge), "--uhf", str(job.uhf),
           "-P", str(xcfg["threads_per_job"])]
    if job.opt and not single_atom:
        cmd += ["--opt", xcfg["opt_level"]]
    if xcfg.get("solvent"):
        cmd += ["--alpb", xcfg["solvent"]]
    env = dict(os.environ, OMP_NUM_THREADS=str(xcfg["threads_per_job"]),
               OMP_STACKSIZE="2G", MKL_NUM_THREADS=str(xcfg["threads_per_job"]))
    try:
        with open(wd / "xtb.out", "w") as fh:
            subprocess.run(cmd, cwd=wd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                           timeout=xcfg["timeout_s"], check=False)
    except subprocess.TimeoutExpired:
        return XtbResult(wd, False, None, error="timeout")
    except FileNotFoundError:
        raise RuntimeError(f"xtb executable not found: {xcfg['executable']}")
    res = _parse(wd, job.opt and not single_atom)
    last = next((wd / f for f in ("xtblast.xyz", "xtbopt.xyz") if (wd / f).exists()), None)
    if not res.ok and res.error == "optimization not converged" and last is not None:
        # one restart from the last geometry with a larger cycle budget
        last.replace(wd / "restart.xyz")
        try:
            with open(wd / "xtb.out", "w") as fh:
                subprocess.run([c if c != "input.xyz" else "restart.xyz" for c in cmd]
                               + ["--cycles", str(max(500, 10 * len(job.symbols)))],
                               cwd=wd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                               timeout=xcfg["timeout_s"], check=False)
        except subprocess.TimeoutExpired:
            return XtbResult(wd, False, None, error="timeout")
        res = _parse(wd, True)
    if not res.ok:
        log.warning("xtb failed in %s: %s", wd, res.error)
    return res


def run_many(jobs: list[XtbJob], xcfg: dict, label: str = "") -> list[XtbResult]:
    if not jobs:
        return []
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=xcfg["parallel_jobs"]) as ex:
        futs = {ex.submit(run_job, j, xcfg): i for i, j in enumerate(jobs)}
        for k, f in enumerate(as_completed(futs), 1):
            results[futs[f]] = f.result()
            if k % 10 == 0 or k == len(jobs):
                log.info("%s xtb jobs: %d/%d done", label, k, len(jobs))
    return results
