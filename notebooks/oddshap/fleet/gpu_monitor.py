"""Live dashboard for the distributed GPU reproduction run.

Reads ``fleet_status.json`` (written by gpu_fleet.py) and renders a four-panel terminal
dashboard, refreshing in place. Everything is derived from the fleet's live state and
measured per-instance timings, so the time forecasts sharpen as the run proceeds.

    python notebooks/oddshap/fleet/gpu_monitor.py                       # live, refresh 5s
    python notebooks/oddshap/fleet/gpu_monitor.py --once                # one snapshot
    python notebooks/oddshap/fleet/gpu_monitor.py --cost 1.5            # yuan per GPU-hour
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path


def fmt_dur(s: float) -> str:
    s = int(max(0, s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def bar(pct: float, width: int = 24) -> str:
    n = int(pct / 100 * width)
    return "█" * n + "░" * (width - n)


def render(snap: dict, cost_per_gpu_h: float | None) -> str:
    now = time.time()
    L = []
    el = snap["elapsed_s"]
    L.append("═" * 66)
    L.append(f" GPU reproduction fleet · {datetime.now().strftime('%H:%M:%S')} · "
             f"elapsed {fmt_dur(el)}")
    L.append("═" * 66)

    # ---- ① global progress ----
    done, tot = snap["done"], snap["jobs_total"]
    L.append(f"PROGRESS  {done}/{tot}  {bar(snap['pct'])} {snap['pct']:5.1f}%   "
             f"failed={snap['failed_final']}")
    cells = snap["done_by_cell"]
    per_cell_total = tot // max(1, len(cells)) if cells else tot
    cellline = "   ".join(f"{k}:{v}" for k, v in sorted(cells.items()))
    if cellline:
        L.append(f"          {cellline}")

    # ---- ② detailed time expectations ----
    L.append("─" * 66)
    L.append("TIME  throughput %.1f inst/min · median/instance: %s" % (
        snap["throughput_per_min"],
        "  ".join(f"{vf}={fmt_dur(s)}" for vf, s in snap["vf_median_s"].items())))
    eta_s = snap["eta_s"]
    finish = datetime.now() + timedelta(seconds=eta_s)
    L.append(f"      remaining ≈ {fmt_dur(eta_s)}   →  finish ~ {finish.strftime('%H:%M')}")
    for vf, s in snap["eta_by_vf_s"].items():
        f2 = datetime.now() + timedelta(seconds=s)
        L.append(f"        {vf:<11} left ≈ {fmt_dur(s):<7} → {f2.strftime('%H:%M')}")
    if cost_per_gpu_h:
        gh = snap["gpu_hours"]
        proj_gh = gh + eta_s * max(1, len([m for m in snap['machines'] if m['status'] != 'offline'])) / 3600
        L.append(f"COST  spent {gh:.2f} GPU-h ≈ ¥{gh*cost_per_gpu_h:.1f}  ·  "
                 f"projected ~¥{proj_gh*cost_per_gpu_h:.1f}")

    # ---- ③ per-machine live (each machine runs `slots` concurrent instances) ----
    L.append("─" * 66)
    L.append(f"{'MACHINE':<20}{'STATE':<7}{'GPU util/mem/temp':<20}{'active jobs (elapsed)':<26}")
    for m in snap["machines"]:
        stale = "⚠" if m["last_beat"] and now - m["last_beat"] > 180 else " "
        gpu = f"{m['gpu_util']:.0f}% {m['gpu_mem_used']:.0f}/{m['gpu_mem_total']:.0f}M {m['gpu_temp']:.0f}C" \
            if m["gpu_mem_total"] else "-"
        active = m.get("active", {})
        jobs = []
        for slot in sorted(active, key=lambda s: int(s)):
            info = active[slot]
            on = now - info["started"] if info.get("started") else 0
            jobs.append(f"{info['job'].split('err')[0][:16]}({fmt_dur(on)})")
        jobstr = " ".join(jobs)[:26] if jobs else "-"
        L.append(f"{stale}{m['name'][:19]:<19}{m['status']:<7}{gpu:<20}{jobstr:<26}"
                 f" ok={m['done']} f={m['failed']}")

    L.append("═" * 66)
    return "\n".join(L)


def main():
    try:  # box-drawing chars need utf-8; Windows consoles default to a legacy codepage
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="notebooks/oddshap/fleet/out/fleet_status.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--cost", type=float, default=None, help="yuan per GPU-hour for cost tracking")
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()
    p = Path(a.status)
    while True:
        if not p.exists():
            print(f"waiting for {p} …")
        else:
            try:
                snap = json.loads(p.read_text())
                out = render(snap, a.cost)
                if not a.once:
                    print("\033[2J\033[H", end="")   # clear + home
                print(out)
            except (json.JSONDecodeError, KeyError):
                pass
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
