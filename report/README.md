# Report

The group report. Skeleton only: the structure, the setup, and the configuration tables are
in place; the prose is `TODO` and the placeholder text is `\lipsum`, which is there so the
document compiles and paginates while it is being written. Every `\lipsum` line is a
paragraph somebody still has to write.

## Building

```bash
latexmk -pdf main.tex      # or: pdflatex main && biber main && pdflatex main x2
```

Needs a TeX distribution with `biber`. `cleanthesis.sty` is vendored here, so nothing has
to be installed for the chair's look.

## Structure

| File | Section |
|---|---|
| `main.tex` | document skeleton — title, `\input`s, bibliography |
| `report-setup.tex` | packages, the chair's style, report metadata (**edit the author list here**) |
| `content/01-introduction.tex` | 1. Abstract and introduction |
| `content/02-benchmark.tex` | 2. Cross-method benchmark |
| `content/03-oddshap.tex` | 3. OddSHAP |
| `content/04-leverageshap.tex` | 4. LeverageSHAP |
| `content/05-polyshap.tex` | 5. PolySHAP |
| `references.bib` | pulled from `docs/source/references.bib`, so keys match the codebase |
| `figures/` | put exported figures here |

Each approximator section has the same two subsections — **Summary**, then
**Reproduction and Discussion** — and every reproduction opens with a configuration table
before any numbers. That order is deliberate: the three papers do not share a value function, so a
configuration read after the results is a configuration read too late.

## Where the template came from

The chair's thesis template (`template-aiml-latex-thesis-main`, `cleanthesis.sty`), reduced
to a report:

* kept `scrreprt` and `cleanthesis`, which is what the chair's template uses and what
  `cleanthesis` is built for — the five top-level parts are `\chapter`s;
* kept the chair's title page, its logos and its title colour, minus the reviewer block a
  report does not have;
* single-sided, `open=any`, no `\cleardoublepage`;
* `BCOR=0mm`. `cleanthesis` sets a 25 mm binding correction, which is right for a printed
  and bound thesis but on screen only pushes every page to the right;
* `\chapterpagestyle` set to `scrheadings`. `cleanthesis` moves the page number into the
  footer but KOMA's own centred one survives in `plain`, which KOMA uses for
  chapter-opening pages — so those pages printed the number twice;
* dropped the thesis front and back matter: cover page, declaration, acknowledgement,
  colophon, list of figures/tables.

## Writing notes

* **Two setups, two questions.** Section 2 puts every method on one shared value function
  so they can be compared with each other. Sections 3–5 each use *their own paper's* value
  function, because that is the only way to test that paper's claim. Absolute errors are
  therefore not comparable across the two, and `01-introduction.tex` says so up front —
  keep that, or a reader will read the tables as contradicting each other.
* **Numbers come from committed results, not from memory.** The benchmark section draws on
  `benchmark/results/lmu_full_sweep_20260717/` (its README records the exact command, the
  code revision and the machine); the OddSHAP section draws on `notebooks/oddshap/`.
* **Sections 4 and 5 are their authors' to write.** The tables there are `TODO` on purpose:
  filling them in from guesswork is how a wrong configuration ends up in a report.
