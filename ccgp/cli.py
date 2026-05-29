"""Minimal command-line interface for ccgp.

Experiments are driven by ``python experiments/run.py {hpo,main,batch,splits,all}``;
this CLI exposes quick utilities.
"""
from __future__ import annotations

import argparse
import json

from . import __version__


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ccgp",
                                 description="Correlation-Consistent Genomic Prediction")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("version", help="print version")
    sub.add_parser("datasets", help="list available public datasets")
    sub.add_parser("losses", help="list training losses")
    sub.add_parser("verify", help="run the numerical equivalence checks (Experiment A)")
    args = ap.parse_args(argv)

    if args.cmd in (None, "version"):
        print(__version__)
    elif args.cmd == "datasets":
        from .data import EASYGESE_SPECIES, SOYNAM_TRAITS
        print("EasyGeSe (10 species):", ", ".join(EASYGESE_SPECIES))
        print("CIMMYT wheat: 4 environments (env1..env4)")
        print("SoyNAM traits:", ", ".join(SOYNAM_TRAITS))
    elif args.cmd == "losses":
        from .losses import available_losses
        print(", ".join(available_losses()))
    elif args.cmd == "verify":
        from experiments.exp_a_numerical import main as expa
        expa()


if __name__ == "__main__":
    main()
