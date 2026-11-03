"""Command line entry point.

    python -m tfa_eads prepare     -c config.yaml
    python -m tfa_eads xtb         -c config.yaml [--ids 413436 1296] [--limit 10]
    python -m tfa_eads dft-setup   -c config.yaml [--ids ...]
    python -m tfa_eads dft-collect -c config.yaml
"""
from __future__ import annotations

import argparse
import logging


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tfa_eads", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "xtb", "dft-setup", "dft-collect"])
    ap.add_argument("-c", "--config", default=None, help="YAML config (defaults used if omitted)")
    ap.add_argument("--ids", nargs="*", default=None, help="only these molecule ids")
    ap.add_argument("--limit", type=int, default=None, help="only the first N molecules (testing)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    from .config import load_config
    cfg = load_config(args.config)

    if args.command == "prepare":
        from .ingest import run_prepare
        run_prepare(cfg)
    elif args.command == "xtb":
        from .workflow import run_xtb_tier
        run_xtb_tier(cfg, ids=args.ids, limit=args.limit)
    elif args.command == "dft-setup":
        from .dft import run_dft_setup
        run_dft_setup(cfg, ids=args.ids)
    elif args.command == "dft-collect":
        from .dft import run_dft_collect
        run_dft_collect(cfg)


if __name__ == "__main__":
    main()
