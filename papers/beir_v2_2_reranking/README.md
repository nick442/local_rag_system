# BEIR v2.2 Reranking Paper

This folder contains:
- `paper.md`: Markdown version of the paper
- `main.tex`: IEEE conference template (LaTeX)
- `references.bib`: Bibliography stub (IEEEtran)

## Build (LaTeX)

```bash
# from repo root
cd papers/beir_v2_2_reranking
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `IEEEtran` or other packages are missing, install a full TeX distribution (e.g., TeX Live).

## Notes
- Metrics and methods correspond to `experiments/reranking/FINAL_REPORT_v2_2_BEIR.md` and `experiments/reranking/results/summary.md`.
- Replace bibliography stubs with finalized citations as needed.
