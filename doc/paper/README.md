# Paper artifacts

This folder contains the LaTeX source for the project paper and the empirical artifacts (plots/tables) generated from the batch benchmark.

## Compile

From this folder:

```bash
make
```

If you prefer manual compilation:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Images

All figures are expected under:

```
doc/paper/images/
```

This repo already generates benchmark plots into that folder. For Langfuse screenshots, please add:

- `doc/paper/images/langfuse_graphrag_trace.png`
- `doc/paper/images/langfuse_rau_trace.png`

The paper will render placeholders if screenshots are missing.

