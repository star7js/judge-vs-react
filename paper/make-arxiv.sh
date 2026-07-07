#!/usr/bin/env bash
# Assemble the arXiv submission bundle: judge-vs-react.tex + the 6 figures it
# references, flattened into one directory (arXiv extracts to a single dir, so
# the repo's ../eval/pit/figures/ paths won't resolve there). The .tex uses
# \graphicspath to find figures/ (bundle) or ../eval/pit/figures/ (in-repo), so
# the SAME source compiles in both places.
#
# Output: judge-vs-react-arxiv.tar.gz  (upload this to arXiv, source-only).
set -euo pipefail
cd "$(dirname "$0")"

FIGS=(fig_timing fig_latency fig_flip fig_repeat fig_perception fig_aim)

rm -rf arxiv && mkdir -p arxiv/figures
command cp judge-vs-react.tex arxiv/judge-vs-react.tex
for f in "${FIGS[@]}"; do
  command cp "../eval/pit/figures/$f.png" "arxiv/figures/$f.png"
done

# Verify it compiles self-contained (tectonic; arXiv uses pdflatex, which the
# \pdftexversion-guarded \pdfoutput=1 in the .tex targets). Then drop the PDF —
# arXiv wants source only.
if command -v tectonic >/dev/null 2>&1; then
  ( cd arxiv && tectonic judge-vs-react.tex >/dev/null 2>&1 && rm -f judge-vs-react.pdf )
  echo "self-contained build OK"
fi

tar czf judge-vs-react-arxiv.tar.gz -C arxiv judge-vs-react.tex figures
echo "wrote judge-vs-react-arxiv.tar.gz:"
tar tzf judge-vs-react-arxiv.tar.gz
