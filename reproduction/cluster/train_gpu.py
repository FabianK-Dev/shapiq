"""GPU value-function training for the OddSHAP reproduction (ViT16, DistilBERT).

For each deep-learning value function, computes the exact Shapley ground truth
(``ExactComputer`` over all 2**d coalitions) once per instance and reuses it for:

  * Table 1 / Table 3   -- MSE at budget = 100*d
  * Figure 2            -- MSE vs budget grid
  * Figure 4 / Figure 11-- eta ablation at m in {5000, 10000, 20000}

Emits tidy lines the aggregator turns into CSVs:
    PARTIAL_T1  <vf> <est>  <idx> <budget> <mse>
    PARTIAL_F2  <vf> <est>  <idx> <budget> <mse>
    PARTIAL_ETA <vf> <eta|base> <idx> <budget> <mse>

Sliced by [--start,--end) so 30 instances run as parallel Slurm array shards.

Usage:  python -m reproduction.cluster.train_gpu --vf vit16 --variant v522_merged \
            --start 0 --end 3
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from shapiq.game_theory.exact import ExactComputer

from reproduction.core.constants import ESTIMATORS as EST
from reproduction.core.constants import ETA_BUDGETS, ETAS, VARIANT_CHOICES
from reproduction.core.harness import interaction_free_oddshap, load_oddshap, log_budgets, make_estimator, sfv

N_INST = 30
REPO = os.path.expanduser("~/oddshap_reproduction")
EXCERPT_TOKENS = 14
REVIEWS = [
    "This film is an absolute masterpiece with stunning visuals and a deeply moving story.",
    "What a complete waste of time, the plot made no sense and the acting was wooden.",
    "I loved every minute of this movie, the direction was brilliant and the cast superb.",
    "Painfully boring from start to finish, I nearly fell asleep before the first act ended.",
    "An unforgettable experience, beautifully shot and emotionally powerful in every scene.",
    "The script was lazy, the jokes fell flat, and the characters were utterly forgettable.",
    "A triumph of modern cinema, gripping, intelligent, and genuinely thrilling throughout.",
    "Terrible pacing and a predictable ending ruined what could have been a decent thriller.",
    "Heartwarming and funny, this is easily one of the best films I have seen this year.",
    "The special effects were laughable and the dialogue made me cringe in my seat.",
    "A stunning performance by the lead actor carries this powerful and affecting drama.",
    "Dull, lifeless, and far too long, this movie tested my patience at every turn.",
    "Visually breathtaking with a haunting score that elevates an already remarkable work.",
    "I cannot recommend this disaster, it was a confusing and pretentious mess throughout.",
    "Charming, witty, and surprisingly touching, this little gem deserves a wider audience.",
    "The plot holes were enormous and the ending felt rushed and completely unearned.",
    "A bold and original vision that rewards patient viewers with one of the finest endings.",
    "Cliched, derivative, and forgettable, nothing here you have not already seen before.",
    "Brilliantly acted and tightly directed, this gripping thriller kept me on edge.",
    "An incoherent jumble of half-baked ideas that never once manages to engage.",
    "Funny, warm, and genuinely uplifting, this feel-good movie is a joy throughout.",
    "The worst film I have endured in years, badly written and even more badly performed.",
    "A masterclass in tension and atmosphere, every frame crafted with obvious care.",
    "Tedious and self-indulgent, the director fell in love with his own boring footage.",
    "Genuinely scary and superbly paced, this horror film delivers chills without cheap tricks.",
    "Flat performances and a meandering plot make this one of the most disappointing sequels.",
    "A delightful surprise, smart and heartfelt with a wonderful ensemble cast.",
    "Overlong, overwrought, and utterly humorless, this bloated epic collapses.",
    "Inventive and thrilling, it reinvents the genre with style, wit, and emotional depth.",
    "A forgettable cash grab with wooden acting, recycled jokes, and no reason to exist.",
]


def make(name, n, variant):
    """Estimator by name — delegates to the shared harness (single source of truth)."""
    return make_estimator(name, n, oddshap_variant=variant)


# single-feature Shapley vector — the shared harness helper
singles = sfv


def _require_gpu() -> None:
    """Fail loudly if no CUDA device is visible, instead of silently running on CPU
    (DistilBERT/ViT exact GT on CPU is prohibitively slow)."""
    try:
        import torch

        if not torch.cuda.is_available():
            msg = "no CUDA device visible — the GPU value functions must run on a GPU node"
            raise RuntimeError(msg)
    except ImportError:
        pass  # torch resolved lazily by the model builders; skip the check if absent here


def vit_builder():
    from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier
    imgs = sorted(glob.glob(f"{REPO}/src/shapiq_games/benchmark/imagenet_examples/*.JPEG"))[:N_INST]

    def build(i):
        g = ImageClassifier(model_name="vit_16_patches", x_explain_path=imgs[i], normalize=True, verbose=False)
        return g, g.n_players
    return build


def distilbert_builder():
    from transformers import AutoTokenizer
    from shapiq_games.benchmark.local_xai.benchmark_language import SentimentAnalysis
    tok = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    reviews = []
    for text in REVIEWS:
        ids = tok(text)["input_ids"][1:1 + EXCERPT_TOKENS]
        if len(ids) == EXCERPT_TOKENS:
            reviews.append(tok.decode(ids))
        if len(reviews) >= N_INST:
            break
    if len(reviews) < N_INST:
        msg = f"only {len(reviews)} reviews reached {EXCERPT_TOKENS} tokens; need {N_INST}"
        raise RuntimeError(msg)

    def build(i):
        g = SentimentAnalysis(input_text=reviews[i], device=0, verbose=False)
        return g, g.n_players
    return build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", required=True, choices=["vit16", "distilbert"])
    ap.add_argument("--variant", default="v522_merged", choices=VARIANT_CHOICES)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_INST)
    ap.add_argument("--experiments", nargs="+", default=["table1", "fig2", "eta"],
                    choices=["table1", "fig2", "eta"],
                    help="which experiments to run per instance (split for short backfill jobs)")
    args = ap.parse_args()
    _require_gpu()
    exp = set(args.experiments)
    build = vit_builder() if args.vf == "vit16" else distilbert_builder()

    for i in range(max(0, args.start), min(args.end, N_INST)):
        game, n = build(i)
        gt = singles(ExactComputer(game=game, n_players=n)(index="SV"), n)
        # Table 1
        b1 = max(n + 1, 100 * n)
        if "table1" in exp:
            for e in EST:
                try:
                    mse = float(np.mean((singles(make(e, n, args.variant).approximate(b1, game), n) - gt) ** 2))
                    print(f"PARTIAL_T1 {args.vf} {e} {i} {b1} {mse:.6e}", flush=True)
                except (ValueError, RuntimeError):
                    pass
        # Figure 2 (MSE vs budget) + Figure 5 (runtime vs budget)
        for b in (log_budgets(n) if "fig2" in exp else []):
            for e in EST:
                try:
                    t0 = time.perf_counter()
                    iv = make(e, n, args.variant).approximate(b, game)
                    dt = time.perf_counter() - t0
                    mse = float(np.mean((singles(iv, n) - gt) ** 2))
                    print(f"PARTIAL_F2 {args.vf} {e} {i} {b} {mse:.6e}", flush=True)
                    print(f"PARTIAL_RT {args.vf} {e} {i} {b} {dt:.6e}", flush=True)
                except (ValueError, RuntimeError):
                    pass
        # Figure 4 / 11 — eta at three budgets
        for budget in (ETA_BUDGETS if "eta" in exp else []):
            for et in ETAS:
                try:
                    mse = float(np.mean((singles(load_oddshap(args.variant)(n=n, random_state=0, interaction_factor=et).approximate(budget, game), n) - gt) ** 2))
                    print(f"PARTIAL_ETA {args.vf} {et} {i} {budget} {mse:.6e}", flush=True)
                except (ValueError, RuntimeError):
                    pass
            try:
                iv0 = interaction_free_oddshap(n, oddshap_variant=args.variant).approximate(budget, game)
                mse = float(np.mean((singles(iv0, n) - gt) ** 2))
                print(f"PARTIAL_ETA {args.vf} base {i} {budget} {mse:.6e}", flush=True)
            except (ValueError, RuntimeError):
                pass
        print(f"INSTANCE_DONE {args.vf} {args.variant} {i}", flush=True)
    print(f"DONE_SHARD {args.vf} {args.variant} {args.start} {args.end}", flush=True)


if __name__ == "__main__":
    main()
