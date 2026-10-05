"""Build and package the current G3 initial submission without rerunning experiments."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "submission_g3_2026-10-05"
UPLOAD = DEST / "upload"
VALIDATION = DEST / "validation"


def run(args: list[str], cwd: Path, log: Path) -> None:
    with log.open("w") as handle:
        subprocess.run(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def build(directory: Path, stem: str) -> None:
    for step, args in enumerate([
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem],
        ["bibtex", stem],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem],
    ], start=1):
        run(args, directory, VALIDATION / f"build_{stem}_{step}.txt")


def archive(destination: Path, paths: list[Path], note: str) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("READ_ME.txt", note)
        for path in sorted(set(paths)):
            if path.is_file():
                bundle.write(path, path.relative_to(ROOT))


def main() -> None:
    for directory in [UPLOAD, VALIDATION, DEST / "alternatives", DEST / "source"]:
        directory.mkdir(parents=True, exist_ok=True)
    for directory, stem in [(ROOT / "paper/g3", "main_g3"), (ROOT / "paper", "main"),
                            (ROOT / "paper", "supplement")]:
        build(directory, stem)
    shutil.copy2(ROOT / "paper/g3/main_g3.pdf", UPLOAD / "Manuscript_G3.pdf")
    shutil.copy2(ROOT / "paper/main.pdf", DEST / "alternatives/Manuscript_single_column.pdf")
    shutil.copy2(ROOT / "paper/supplement.pdf", UPLOAD / "Supplementary File 1.pdf")
    for index, path in enumerate(sorted((ROOT / "figures").glob("fig[1-6]_*.pdf")), start=1):
        shutil.copy2(path, UPLOAD / f"Figure {index}.pdf")
    code = [path for folder in ["ccgp", "experiments", "analysis"]
            for path in (ROOT / folder).rglob("*.py")]
    code += [ROOT / name for name in ["pyproject.toml", "README.md", "LICENSE", "reproduce.sh"]]
    code += [ROOT / "figures/make_figures.py", ROOT / "scripts/prepare_g3_submission.py"]
    archive(UPLOAD / "Supplementary File 2.zip", code,
            "Analysis and simulation code. Extract preserving directories. Install with\n"
            "python -m pip install -e '.[dev]'. See README.md for commands and public data sources.\n"
            "Full numerical outputs are in the repository; this ZIP contains no private data.\n")
    data = list((ROOT / "figures/source_data").glob("*.csv"))
    data += list((ROOT / "results/analysis_final").glob("*.csv"))
    data += [ROOT / "figures/text_alternatives.md"]
    archive(UPLOAD / "Supplementary File 3.zip", data,
            "Figure source tables, analysis tables, and figure text alternatives.\n"
            "Source CSV files use the original repository paths. See text_alternatives.md\n"
            "for the conclusions conveyed by Figures 1–6. NDCG@10 refers to the top 10%.\n")
    paper = [ROOT / f"paper/{name}" for name in ["main.tex", "abstract_body.tex", "sec_results.tex",
             "discussion_body.tex", "supplement.tex", "refs.bib", "cover_letter.md"]]
    paper += list((ROOT / "paper/figures").glob("*.pdf"))
    paper += list((ROOT / "paper/supp").glob("tabS_final*.tex"))
    paper += [p for p in (ROOT / "paper/g3").rglob("*") if p.suffix in
              {".tex", ".bib", ".bst", ".cls", ".sty", ".pdf"} and p.name != "main_g3.pdf"
              and "sorghum" not in p.name]
    archive(DEST / "source/Manuscript_LaTeX.zip", paper,
            "Editable manuscript sources. Extract preserving paper/ and paper/g3/.\n"
            "cd paper/g3; pdflatex main_g3; bibtex main_g3; pdflatex main_g3; pdflatex main_g3\n"
            "Supplement: cd paper; pdflatex supplement; bibtex supplement; pdflatex supplement twice.\n")
    frozen = [ROOT / name for name in [
        "results/results_main_campaign.parquet", "results/results_loss_specific_wheat.parquet",
        "results/hpo_campaign.json", "results/hpo_loss_specific_wheat.json", "results/exp_a.json",
        "results/analysis_final/provenance.json",
    ]]
    frozen += list((ROOT / "results/analysis_final").glob("*.csv"))
    frozen += list((ROOT / "figures/source_data").glob("*.csv"))
    archive(DEST / "source/Reproducibility_frozen_results.zip", frozen,
            "Frozen scientific results and selected hyperparameters. This archive is for a public\n"
            "repository deposit, not direct journal upload (larger than the supplement size limit).\n"
            "Use Supplementary File 2.zip for code and Manuscript_LaTeX.zip for paper sources.\n")
    letter = ROOT / "paper/cover_letter.md"
    letter_text = letter.read_text().split("\n", 1)[1].strip() + "\n"
    (DEST / "cover_letter.txt").write_text(letter_text)
    (DEST / "cover_letter.md").write_text(letter.read_text())
    run(["pandoc", "-f", "markdown+hard_line_breaks", str(letter), "-o", str(UPLOAD / "Cover_letter.pdf"), "--pdf-engine=xelatex",
         "-V", "geometry:margin=1in", "-V", "fontsize=11pt"], ROOT, VALIDATION / "build_cover_letter.txt")
    run(["pandoc", "-f", "markdown+hard_line_breaks", str(letter), "-o", str(DEST / "source/Cover_letter.docx")], ROOT,
        VALIDATION / "build_cover_letter_docx.txt")
    abstract_tex = (ROOT / "paper/abstract_body.tex").read_text()
    abstract = abstract_tex.replace(r"$\mathrm{MSE}(z_y,z_{\hat y})=2(1-r)$",
                                    "MSE(z_y, z_prediction) = 2(1 − r)")
    abstract = abstract.replace(r"\%", "%").replace("--", "–").replace("$", "")
    abstract = " ".join(abstract.split()).replace("NDCG@10", "NDCG@10")
    summary = " ".join((ROOT / "paper/g3/article_summary.md").read_text().split())
    (DEST / "abstract.txt").write_text(abstract + "\n")
    (DEST / "article_summary.txt").write_text(summary + "\n")
    counts = {"abstract_words_whitespace": len(abstract.split()), "summary_words_whitespace": len(summary.split()),
              "running_title_characters": len("Metric-consistent genomic prediction")}
    assert counts["abstract_words_whitespace"] <= 250
    assert counts["summary_words_whitespace"] == 100
    assert counts["running_title_characters"] <= 50
    for path in UPLOAD.glob("Supplementary*"):
        assert path.stat().st_size < 2_000_000, path
    assert sum(p.stat().st_size for p in UPLOAD.glob("Supplementary*")) < 10_000_000
    (VALIDATION / "word_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    for stem in ["main_g3", "main", "supplement"]:
        final_log = (VALIDATION / f"build_{stem}_4.txt").read_text(errors="replace")
        assert not re.search(r"undefined|Overfull|Rerun to get", final_log), stem
    paths = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (DEST / "SHA256SUMS.txt").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DEST)}\n" for p in paths))
    print(json.dumps(counts, indent=2))
    for path in sorted(UPLOAD.iterdir()):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
