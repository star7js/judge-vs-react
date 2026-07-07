# Paper: *Judge vs. React*

Dissociating perception from timing in VLM reflex control.

| File | What it is |
|---|---|
| `judge-vs-react.pdf` | The arXiv-ready preprint (built artifact, committed for sharing). |
| `judge-vs-react.tex` | LaTeX source. |
| `judge-vs-react.md` | The working markdown draft (still has a few scaffold notes; the `.tex`/`.pdf` are the clean version). |
| `blog.md` | The accessible blog writeup with the hero demo clip. |

## Build the PDF

Figures are pulled from `../eval/pit/figures/` (regenerate them with
`python -m eval.pit.analyze`). Uses [tectonic](https://tectonic-typesetting.github.io/)
(self-contained; downloads packages on first run):

```bash
cd paper
tectonic judge-vs-react.tex     # -> judge-vs-react.pdf
```

Any LaTeX engine works too (`pdflatex judge-vs-react.tex`), given the graphics
packages.

## Submit to arXiv

`judge-vs-react-arxiv.tar.gz` is the ready-to-upload, source-only bundle:
`judge-vs-react.tex` + a flat `figures/` dir (arXiv extracts to one directory,
so the in-repo `../eval/pit/figures/` paths won't resolve there — the `.tex`
uses `\graphicspath` to find figures in either location, and a
`\pdftexversion`-guarded `\pdfoutput=1` so arXiv builds it with pdflatex).

1. Upload `judge-vs-react-arxiv.tar.gz` at arxiv.org (New Submission → upload the
   tarball). arXiv compiles the source itself; don't include the PDF.
2. Suggested category: `cs.AI` (cross-list `cs.LG`, `cs.CV`).

Rebuild the bundle any time with `./make-arxiv.sh` (regenerates the tarball from
the current `.tex` and figures; verifies a self-contained compile).
