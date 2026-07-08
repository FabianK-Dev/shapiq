"""Robust distributed GPU launcher — detached shards + log polling.

Lesson from the first fleet: piping ``train_gpu | tee`` back through the SSH channel
couples every remote job to the dispatcher's stdout consumption. When the dispatcher died,
tee blocked on the broken stdout pipe and every job hung. This launcher instead starts each
shard **detached** (``nohup … > shard.log &``) so jobs are independent of this process — if
the launcher dies, the GPUs keep working and re-running the poll just resumes tracking.

Work is split into shards of ``SHARD`` instances per (vf, variant); shards are spread
round-robin across the machines, giving ~6 concurrent processes per box. Each shard writes
its ``PARTIAL_*`` lines to ``reproduction/data/shard_<vf>_<variant>_<s>_<e>.log`` on the
remote; the poller reads those to report progress and, at the end, pulls them all back.

    python -m reproduction.fleet.gpu_run --machines reproduction/cluster/machines.txt \
        --experiments table1 fig2 eta            # launch + poll to completion
    python -m reproduction.fleet.gpu_run --poll-only   # just resume polling
    python -m reproduction.fleet.gpu_run --pull        # pull all shard logs locally
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import paramiko

REMOTE_REPO = "/root/oddshap_reproduction"
REMOTE_PY = "/root/miniconda3/bin/python"
CELLS = [("distilbert", "v522_merged"), ("distilbert", "v560_improved"),
         ("vit16", "v522_merged"), ("vit16", "v560_improved")]
N_INST = 30
SHARD = 5                       # instances per detached process
OUT = Path("reproduction/fleet/out")


def parse_machines(path):
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        h, p, u, pw = ln.split(":", 3)
        out.append({"name": f"{h}:{p}", "host": h, "port": int(p), "user": u, "pw": pw})
    return out


def connect(m):
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(m["host"], port=m["port"], username=m["user"], password=m["pw"], timeout=30)
    return c


def run(c, cmd, t=60):
    _i, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()


def shard_name(vf, var, s, e):
    return f"shard_{vf}_{var}_{s}_{e}"


def build_shards(cells=CELLS):
    shards = []
    for vf, var in cells:
        for s in range(0, N_INST, SHARD):
            shards.append((vf, var, s, min(s + SHARD, N_INST)))
    return shards


def assign(shards, machines):
    """Round-robin so each box gets a mix of light (DistilBERT) and heavy (ViT) shards."""
    a = {m["name"]: [] for m in machines}
    for i, sh in enumerate(shards):
        a[machines[i % len(machines)]["name"]].append(sh)
    return a


def launch(machines, assignment, experiments, extra=""):
    exp = " ".join(experiments)
    for m in machines:
        c = connect(m)
        run(c, "pkill -f train_gpu 2>/dev/null; sleep 2")   # clear any stuck jobs
        run(c, f"mkdir -p {REMOTE_REPO}/reproduction/data")
        for vf, var, s, e in assignment[m["name"]]:
            log = f"reproduction/data/{shard_name(vf, var, s, e)}.log"
            # setsid + redirect all three streams away from the channel so exec_command
            # returns immediately and the job survives this connection closing.
            cmd = (f"cd {REMOTE_REPO}; "
                   f"HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                   f"TMPDIR=/tmp OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 "
                   f"setsid {REMOTE_PY} -m reproduction.cluster.train_gpu --vf {vf} --variant {var} "
                   f"--start {s} --end {e} --experiments {exp}{extra} > {log} 2>&1 < /dev/null &")
            c.exec_command(cmd)          # fire and forget — do NOT read (would block on the &)
            time.sleep(0.4)
        print(f"launched {len(assignment[m['name']])} shards on {m['name']}", flush=True)
        time.sleep(1)
        c.close()


def poll(machines, total, interval=20):
    start = None
    while True:
        done = 0
        util_line = []
        alive = 0
        for m in machines:
            try:
                c = connect(m)
                d = int(run(c, f"grep -h INSTANCE_DONE {REMOTE_REPO}/reproduction/data/shard_*.log 2>/dev/null | wc -l") or 0)
                pr = int(run(c, "pgrep -f train_gpu | wc -l") or 0)
                u = run(c, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1")
                done += d; alive += (1 if pr else 0)
                util_line.append(f"{m['name'].split(':')[1]}:{d}done/{pr}p/util {u}")
                c.close()
            except Exception as ex:
                util_line.append(f"{m['name'].split(':')[1]}:ERR {str(ex)[:20]}")
        if start is None:
            start = time.time()
        el = time.time() - start
        rate = done / el * 60 if el > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{total} done · {rate:.1f}/min · "
              f"ETA {eta:.0f}min | " + " | ".join(util_line), flush=True)
        if done >= total:
            print("ALL DONE", flush=True)
            break
        time.sleep(interval)


def pull(machines):
    OUT.mkdir(parents=True, exist_ok=True)
    for m in machines:
        c = connect(m); sf = c.open_sftp()
        d = f"{REMOTE_REPO}/reproduction/data"
        for fn in run(c, f"ls {d}/shard_*.log 2>/dev/null").split():
            base = fn.split("/")[-1]
            sf.get(fn, str(OUT / f"{m['name'].split(':')[1]}_{base}"))
        sf.close(); c.close()
        print(f"pulled from {m['name']}", flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", default="reproduction/cluster/machines.txt")
    ap.add_argument("--experiments", nargs="+", default=["table1", "fig2", "eta"])
    ap.add_argument("--eta-budgets", nargs="+", type=int, default=None)
    ap.add_argument("--fig2-max-budget", type=int, default=None)
    ap.add_argument("--only-vf", default=None, choices=["vit16", "distilbert"],
                    help="restrict to one value function's cells (e.g. re-run just ViT)")
    ap.add_argument("--poll-only", action="store_true")
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args()
    machines = parse_machines(a.machines)
    cells = [c for c in CELLS if a.only_vf is None or c[0] == a.only_vf]
    shards = build_shards(cells)
    total = len(cells) * N_INST
    if a.pull:
        pull(machines); return
    if not a.poll_only:
        assignment = assign(shards, machines)
        extra = ""
        if a.eta_budgets:
            extra += " --eta-budgets " + " ".join(map(str, a.eta_budgets))
        if a.fig2_max_budget:
            extra += f" --fig2-max-budget {a.fig2_max_budget}"
        print(f"{len(shards)} shards over {len(machines)} machines, {total} instances "
              f"(vf={a.only_vf or 'all'}) · opts:{extra or 'full'}", flush=True)
        launch(machines, assignment, a.experiments, extra)
    poll(machines, total)
    pull(machines)


if __name__ == "__main__":
    main()
