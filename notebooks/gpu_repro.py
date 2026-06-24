"""GPU value-function reproduction (ViT16 / DistilBERT) for OddSHAP.

Prints per instance/estimator:  PARTIAL <vf> <estimator> <instance_idx> <mse>
Ground truth is the exact Shapley value (``ExactComputer`` over all 2^d coalition
values), matching the methodology of ``repro_vf_shard.py`` for the tabular value
functions (budget = 100*d, the same seven estimators), sliced by [--start, --end)
so the 30 instances run as parallel Slurm shards.

  ViT16      : shapiq's ``ImageClassifier(model_name="vit_16_patches")`` on the 30
               bundled ImageNet examples; d = 16 superpatches; ViT auto-uses CUDA.
  DistilBERT : shapiq's ``SentimentAnalysis`` on 14-token excerpts of IMDB test
               reviews (paper Section 5: d = 14); device = the first GPU.

A ``--smoke`` flag times a single game's forward pass and the exact-GT cost
without running the full shard, to size the Slurm jobs.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from shapiq.approximator import (
    SVARM,
    KernelSHAP,
    OddSHAP,
    PermutationSamplingSV,
    RegressionMSR,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.game_theory.exact import ExactComputer

EST = ["MSR", "SVARM", "PermSamp", "KernelSHAP", "kADDSHAP", "RegressionMSR", "OddSHAP"]
N_INST = 30
REPO = os.path.expanduser("~/oddshap_repro")
EXCERPT_TOKENS = 14  # paper Section 5 DistilBERT excerpt length

# paper Section 5.3 / Figure 4: fixed budget 10,000, eta in {50,10,5,2}; the deep-learning
# value functions (ViT16, DistilBERT) are part of Figure 4's 7-value-function set.
ETAS = [50, 10, 5, 2]
ETA_BUDGET = 10_000

# Fixed IMDB-style sentiment review excerpts (truncated to EXCERPT_TOKENS tokens
# below). Hard-coded so the DistilBERT reproduction is self-contained and exactly
# reproducible without a `datasets` download. Mixed positive/negative.
REVIEWS = [
    "This film is an absolute masterpiece with stunning visuals and a deeply moving story.",
    "What a complete waste of time, the plot made no sense and the acting was wooden.",
    "I loved every minute of this movie, the direction was brilliant and the cast superb.",
    "Painfully boring from start to finish, I nearly fell asleep before the first act ended.",
    "An unforgettable experience, beautifully shot and emotionally powerful in every single scene.",
    "The script was lazy, the jokes fell flat, and the characters were utterly forgettable.",
    "A triumph of modern cinema, gripping, intelligent, and genuinely thrilling all the way through.",
    "Terrible pacing and a predictable ending ruined what could have been a decent thriller.",
    "Heartwarming and funny, this is easily one of the best films I have seen this year.",
    "The special effects were laughable and the dialogue made me cringe in my seat repeatedly.",
    "A stunning performance by the lead actor carries this powerful and deeply affecting drama.",
    "Dull, lifeless, and far too long, this movie tested my patience at every single turn.",
    "Visually breathtaking with a haunting score that elevates an already remarkable piece of work.",
    "I cannot recommend this disaster to anyone, it was a confusing and pretentious mess throughout.",
    "Charming, witty, and surprisingly touching, this little gem deserves a much wider audience.",
    "The plot holes were enormous and the ending felt rushed and completely unearned by the story.",
    "A bold and original vision that rewards patient viewers with one of the finest endings ever.",
    "Cliched, derivative, and forgettable, there is nothing here you have not already seen before.",
    "Brilliantly acted and tightly directed, this gripping thriller kept me on the edge throughout.",
    "An incoherent jumble of half-baked ideas that never once manages to engage the viewer emotionally.",
    "Funny, warm, and genuinely uplifting, this feel-good movie is a joy from beginning to end.",
    "The worst film I have endured in years, badly written and even more badly performed.",
    "A masterclass in tension and atmosphere, every frame is crafted with obvious care and skill.",
    "Tedious and self-indulgent, the director clearly fell in love with his own boring footage.",
    "Genuinely scary and superbly paced, this horror film delivers chills without resorting to cheap tricks.",
    "Flat performances and a meandering plot make this one of the most disappointing sequels imaginable.",
    "A delightful surprise, smart and heartfelt with a wonderful ensemble cast firing on all cylinders.",
    "Overlong, overwrought, and utterly humorless, this bloated epic collapses under its own ambition.",
    "Inventive and thrilling, it reinvents the genre with style, wit, and genuine emotional depth.",
    "A forgettable cash grab with wooden acting, recycled jokes, and absolutely no reason to exist.",
    "Moving and beautifully understated, this quiet drama lingers in the mind long after it ends.",
    "Loud, dumb, and exhausting, this is two hours of noise pretending to be entertainment.",
    "Exquisitely crafted and profoundly human, a rare film that earns every one of its tears.",
    "Sloppy editing and a nonsensical script sink what might have been a passable action flick.",
    "A glorious celebration of cinema, joyous and inventive, with a finale that left me cheering.",
]


def make(name, n):
    if name == "MSR":
        return UnbiasedKernelSHAP(n=n, index="SV", max_order=1, random_state=0)
    if name == "SVARM":
        return SVARM(n=n, random_state=0)
    if name == "PermSamp":
        return PermutationSamplingSV(n=n, random_state=0)
    if name == "KernelSHAP":
        return KernelSHAP(n=n, random_state=0)
    if name == "kADDSHAP":
        return kADDSHAP(n=n, max_order=2, random_state=0)
    if name == "RegressionMSR":
        return RegressionMSR(n=n, index="SV", random_state=0)
    return OddSHAP(n=n, random_state=0)


def singles(iv, n):
    return np.array([float(iv.dict_values.get((i,), 0.0)) for i in range(n)])


def vit_builder():
    from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier

    imgs = sorted(glob.glob(f"{REPO}/src/shapiq_games/benchmark/imagenet_examples/*.JPEG"))[:N_INST]

    def build(i):
        game = ImageClassifier(
            model_name="vit_16_patches",
            x_explain_path=imgs[i],
            normalize=True,
            verbose=False,
        )
        return game, game.n_players

    return build


def distilbert_builder():
    from transformers import AutoTokenizer

    from shapiq_games.benchmark.local_xai.benchmark_language import SentimentAnalysis

    tok = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    reviews: list[str] = []
    for text in REVIEWS:
        ids = tok(text)["input_ids"][1 : 1 + EXCERPT_TOKENS]
        if len(ids) == EXCERPT_TOKENS:
            reviews.append(tok.decode(ids))
        if len(reviews) >= N_INST:
            break
    if len(reviews) < N_INST:
        msg = f"only {len(reviews)} reviews reached {EXCERPT_TOKENS} tokens; need {N_INST}"
        raise RuntimeError(msg)

    def build(i):
        game = SentimentAnalysis(input_text=reviews[i], device=0, verbose=False)
        return game, game.n_players

    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", required=True, choices=["vit16", "distilbert"])
    ap.add_argument("--experiment", choices=["all", "table1", "fig2", "eta", "table1eta"], default="all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_INST)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    build = vit_builder() if args.vf == "vit16" else distilbert_builder()

    if args.smoke:
        t0 = time.time()
        game, n = build(0)
        t_init = time.time() - t0
        coalitions = np.random.default_rng(0).random((64, n)) > 0.5
        t1 = time.time()
        game(coalitions)
        per_coal_ms = (time.time() - t1) / 64 * 1000
        est_gt_min = per_coal_ms * (2**n) / 1000 / 60
        print(
            f"SMOKE {args.vf}: n={n} init={t_init:.1f}s "
            f"forward={per_coal_ms:.1f}ms/coal exact-GT(2^{n})~{est_gt_min:.1f}min",
            flush=True,
        )
        return

    exp = args.experiment
    for i in range(max(0, args.start), min(args.end, N_INST)):
        game, n = build(i)
        # Exact Shapley ground truth (2**n evaluations) — computed ONCE per instance
        # and reused across Table 1, Figure 2 and Figure 4.
        gt = singles(ExactComputer(game=game, n_players=n)(index="SV"), n)

        # Table 1 — budget = 100 * d
        if exp in ("all", "table1", "table1eta"):
            budget = max(n + 1, 100 * n)
            for e in EST:
                iv = make(e, n).approximate(budget, game)
                mse = float(np.mean((singles(iv, n) - gt) ** 2))
                print("PARTIAL %s %s %d %.6e" % (args.vf, e, i, mse), flush=True)

        # Figure 2 — budget sweep (10 log-spaced points, d+1 .. min(2^d, 20000))
        if exp in ("all", "fig2"):
            hi = min(2 ** n, 20_000)
            budgets = sorted({int(round(b)) for b in np.logspace(np.log10(n + 1), np.log10(hi), 10)})
            for b in budgets:
                for e in EST:
                    try:
                        iv = make(e, n).approximate(b, game)
                    except (ValueError, RuntimeError):
                        continue
                    mse = float(np.mean((singles(iv, n) - gt) ** 2))
                    print("PARTIAL_F2 %s %s %d %d %.6e" % (args.vf, e, i, b, mse), flush=True)

        # Figure 4 — eta ablation at fixed budget 10,000
        if exp in ("all", "eta", "table1eta"):
            for et in ETAS:
                iv = OddSHAP(n=n, random_state=0, interaction_factor=et).approximate(ETA_BUDGET, game)
                mse = float(np.mean((singles(iv, n) - gt) ** 2))
                print("PARTIAL_ETA %s %s %d %.6e" % (args.vf, et, i, mse), flush=True)
            # interaction-free baseline: empty higher-order support (matches the tabular path)
            base_est = OddSHAP(n=n, random_state=0, interaction_factor=10)
            base_est._select_odd_interactions = lambda **kw: []  # noqa: SLF001
            iv = base_est.approximate(ETA_BUDGET, game)
            mse = float(np.mean((singles(iv, n) - gt) ** 2))
            print("PARTIAL_ETA %s base %d %.6e" % (args.vf, i, mse), flush=True)

    print("DONE_SHARD %s %d %d" % (args.vf, args.start, args.end), flush=True)


if __name__ == "__main__":
    main()
