"""Generate tabular-only variants of the reproduction notebooks.

The GPU value functions (ViT16, DistilBERT) need a CUDA fleet; the six tabular value
functions run anywhere. To guarantee a presentable deliverable independent of the GPU
run, this derives ``nb{1,2,3}_tabular.py`` from the originals by emptying ``GPU_VFS`` —
every tabular figure/table is byte-for-byte the same, the deep-model panels simply drop
out. Regenerate whenever the source notebooks change:

    uv run python reproduction/make_tabular_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

# (source, list of (old, new) replacements). The first replacement zeroes GPU_VFS; the
# second stamps the title so a reader knows this is the tabular-only cut.
JOBS = {
    "nb1_reproduction_ours.py": [
        ("TAB_VFS, GPU_VFS = TABULAR_VF_NAMES, GPU_VF_NAMES",
         "TAB_VFS, GPU_VFS = TABULAR_VF_NAMES, []  # tabular-only: ViT16/DistilBERT excluded"),
        ("# # NB1 — OddSHAP paper reproduction with **our merged contribution (PR #522)**",
         "# # NB1 (tabular-only) — OddSHAP reproduction with **our merged contribution (PR #522)**\n"
         "#\n# > **Tabular-only cut.** The six tabular value functions only; the ViT16/DistilBERT\n"
         "# > panels are omitted (they need the GPU fleet). Everything else is identical to NB1."),
    ],
    "nb2_reproduction_author.py": [
        ("TAB_VFS, GPU_VFS = TABULAR_VF_NAMES, GPU_VF_NAMES",
         "TAB_VFS, GPU_VFS = TABULAR_VF_NAMES, []  # tabular-only: ViT16/DistilBERT excluded"),
    ],
    "nb3_comparison.py": [
        ('GPU_VFS = ["vit16", "distilbert"]',
         'GPU_VFS = []  # tabular-only: ViT16/DistilBERT excluded'),
    ],
}


# nb2 is nb1 with the variant switched to the author's PR #560 — derive it so nb1 stays the
# single source of truth for all the shared figure/table code.
NB2_REPLS = [
    ("# # NB1 — OddSHAP paper reproduction with **our merged contribution (PR #522)**",
     "# # NB2 — OddSHAP paper reproduction with the **author's improvement (PR #560)**"),
    ("for Shapley Values*, arXiv:2602.01399) using the OddSHAP implementation **our group\n"
     "# contributed and merged (PR #522)**.",
     "for Shapley Values*, arXiv:2602.01399) using the paper author's follow-up **PR #560**\n"
     "# (relaxed minimum budget + paired-row sampling)."),
    ('VARIANT = os.environ.get("ODDSHAP_VARIANT", "v522_merged")',
     'VARIANT = os.environ.get("ODDSHAP_VARIANT", "v560_improved")'),
    ("plots. Regenerate with `bash reproduction/cluster/submit_all.sh v522_merged`.",
     "plots. Regenerate with `bash reproduction/cluster/submit_all.sh v560_improved`."),
]


def _regen_nb2() -> None:
    text = (HERE / "nb1_reproduction_ours.py").read_text(encoding="utf-8")
    for old, new in NB2_REPLS:
        if old not in text:
            raise SystemExit(f"nb2 pattern not found in nb1:\n  {old[:60]!r}")
        text = text.replace(old, new, 1)
    (HERE / "nb2_reproduction_author.py").write_text(text, encoding="utf-8")
    print("wrote nb2_reproduction_author.py (derived from nb1)")


def main() -> None:
    _regen_nb2()  # keep nb2 in sync with nb1 before deriving the tabular cuts
    for src, repls in JOBS.items():
        text = (HERE / src).read_text(encoding="utf-8")
        for old, new in repls:
            if old not in text:
                raise SystemExit(f"pattern not found in {src}:\n  {old!r}")
            text = text.replace(old, new, 1)
        out = HERE / src.replace(".py", "_tabular.py").replace("_reproduction", "").replace("_comparison", "")
        # normalise names to nb1_tabular.py / nb2_tabular.py / nb3_tabular.py
        stem = src.split("_")[0]
        out = HERE / f"{stem}_tabular.py"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
