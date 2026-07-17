# Notebooks — demos and paper reproductions

Each new approximator has its own directory containing the notebooks that reproduce the
experiments from its paper. Notebooks are committed **with their outputs**, so they can be read
directly without running anything.

| Directory | Approximator | Start with |
|---|---|---|
| [`oddshap/`](oddshap) | **OddSHAP** (Fumagalli et al., 2026) — PRs [#522](https://github.com/mmschlk/shapiq/pull/522), [#560](https://github.com/mmschlk/shapiq/pull/560) | [`oddshap/README.md`](oddshap/README.md) → `oddshap/oddshap_reproduction.ipynb` |
| [`polyshap/`](polyshap) | **PolySHAP** | `polyshap/polyshap_reproduction.ipynb` (plus `polyshap_maxorder_vs_k.ipynb`, `polyshap_true_order.ipynb`) |

The LeverageSHAP reproduction notebooks live on the `submission` branch and will appear here once
that branch is merged; this index should be extended then.

## Conventions

* `<approximator>/<approximator>_<topic>.ipynb` — the rendered notebooks.
* Notebook **sources** are kept as plain `.py` (jupytext) next to them; the `.ipynb` files are
  generated from those, so edit the `.py`.
* Paper reference material and generated inputs live inside each approximator's directory, so a
  notebook directory is self-contained.
