"""Write a compact provenance and SHA256 manifest for deposited artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import statsmodels
import torch


FILES = [
    "results/results_main_campaign.parquet",
    "results/results_loss_specific_wheat.parquet",
    "results/hpo_campaign.json",
    "results/hpo_loss_specific_wheat.json",
    "results/analysis_final/paired_summary.csv",
    "results/analysis_final/mixed_model_interaction_tests.csv",
    "paper/main.pdf",
    "paper/supplement.pdf",
    "paper/g3/main_g3.pdf",
]


def command(*args):
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    generated = []
    for pattern in ["results/analysis_final/*.csv", "figures/source_data/*.csv",
                    "figures/fig*.pdf", "figures/fig*.svg"]:
        generated.extend(str(path) for path in sorted(Path().glob(pattern)))
    files = list(dict.fromkeys(FILES + generated))
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "packages": {
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__, "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "devices": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if torch.cuda.is_available() else []),
        },
        "sha256": {path: digest(path) for path in files if Path(path).exists()},
        "commands": [
            "python analysis/final_analysis.py",
            "python analysis/mixed_model_final.py",
            "python figures/make_figures.py",
            "python analysis/make_final_supplement.py",
        ],
    }
    Path("results/analysis_final/reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
