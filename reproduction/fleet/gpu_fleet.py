"""Distributed GPU reproduction dispatcher.

Runs the deep-learning value functions (ViT16 / DistilBERT) across a fleet of one-GPU
SSH machines. Each unit of work is one (value function, variant, instance) triple; there
are ``2 vf x 2 variant x N`` of them and they are fully independent, so the dispatcher
keeps every machine busy with one instance at a time until the queue drains.

It is a *pull* scheduler: whenever a machine is free it is handed the next pending job,
its stdout (the ``PARTIAL_*`` lines) is streamed back and appended to a central per-VF
log, and the job's timing is recorded. Failed jobs are retried on another machine. A
live ``fleet_status.json`` is written continuously for ``gpu_monitor.py`` to render.

Machines file (``--machines``), one per line: ``host:port:user:password``  (``#`` comments ok).

Usage:
    python -m reproduction.fleet.gpu_fleet --machines machines.txt \
        --variants v522_merged v560_improved --experiments table1 eta --instances 30
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Queue, Empty

import paramiko

REMOTE_REPO = "/root/oddshap_reproduction"        # system disk, so an AutoDL image includes it
REMOTE_PY = "/root/miniconda3/bin/python"
LOCAL_REPRO = Path(__file__).resolve().parents[1]   # local reproduction/ dir to push
GPU_VFS = ["distilbert", "vit16"]     # distilbert first (lighter GT) so results land sooner
# per-instance forward-pass weight, used only for the *initial* ETA before real timings arrive
_VF_WEIGHT = {"distilbert": 1.0, "vit16": 1.6}


@dataclass
class Job:
    vf: str
    variant: str
    inst: int
    attempts: int = 0

    @property
    def key(self) -> str:
        return f"{self.vf}/{self.variant}/{self.inst}"


@dataclass
class MachineState:
    name: str
    status: str = "idle"          # idle | running | offline
    active: dict = field(default_factory=dict)   # slot -> {"job": key, "started": ts}
    slots: int = 1
    done: int = 0
    failed: int = 0
    secs: list[float] = field(default_factory=list)   # per-instance wall times on this machine
    gpu_util: float = 0.0
    gpu_mem_used: float = 0.0
    gpu_mem_total: float = 0.0
    gpu_temp: float = 0.0
    last_beat: float = 0.0


class Fleet:
    def __init__(self, machines, variants, experiments, instances, outdir, hf_mirror=True,
                 per_gpu=1, sync_code=True, eta_budgets=None, cores=48):
        self.machines = machines
        self.experiments = experiments
        self.per_gpu = per_gpu               # concurrent instances per GPU (saturate a small model)
        self.sync_code = sync_code
        self.eta_budgets = eta_budgets       # e.g. [10000] for the reduced Fig4-only run
        self.cores = cores                   # CPU cores per box (to size BLAS threads / process)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.hf_mirror = hf_mirror
        self.q: Queue[Job] = Queue()
        self.jobs_total = 0
        for vf in GPU_VFS:
            for variant in variants:
                for i in range(instances):
                    self.q.put(Job(vf, variant, i)); self.jobs_total += 1
        self.state = {m["name"]: MachineState(m["name"]) for m in machines}
        self.done = 0
        self.failed_final = 0
        self.done_by_cell: dict[str, int] = {}   # "vf/variant" -> count
        self.all_secs: list[float] = []
        self.per_vf_secs: dict[str, list[float]] = {vf: [] for vf in GPU_VFS}
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.status_running = True

    # --- remote helpers ---------------------------------------------------- #
    def _connect(self, m):
        c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(m["host"], port=m["port"], username=m["user"], password=m["password"], timeout=30)
        return c

    def _sync_code(self, m):
        """SFTP the local reproduction/ tree to the machine (GitHub is unreliable on the
        China cloud boxes). Runs once per machine before its workers start."""
        c = self._connect(m)
        sf = c.open_sftp()
        remote_root = f"{REMOTE_REPO}/reproduction"

        def _put(local: Path, remote: str):
            c.exec_command(f"mkdir -p {remote}")[1].read()
            for name in os.listdir(local):
                if name in ("__pycache__", "out", "data"):
                    continue
                lp = local / name; rp = f"{remote}/{name}"
                if lp.is_dir():
                    _put(lp, rp)
                elif name.endswith((".py", ".sh")):
                    sf.put(str(lp), rp)

        _put(LOCAL_REPRO, remote_root)
        sf.close(); c.close()

    def _run_job(self, c, job: Job) -> tuple[bool, str]:
        hf = "HF_ENDPOINT=https://hf-mirror.com " if self.hf_mirror else ""
        exp = " ".join(self.experiments)
        eta = f" --eta-budgets {' '.join(map(str, self.eta_budgets))}" if self.eta_budgets else ""
        # eta's odd-Fourier regression is CPU-heavy; cap BLAS threads so per_gpu processes
        # share the cores without oversubscription (threads = cores / per_gpu).
        thr = max(1, self.cores // self.per_gpu) if self.cores else 4
        threads = (f"OMP_NUM_THREADS={thr} OPENBLAS_NUM_THREADS={thr} "
                   f"MKL_NUM_THREADS={thr} NUMEXPR_NUM_THREADS={thr} ")
        log = f"reproduction/data/gpu_{job.vf}_{job.variant}.log"
        cmd = (f"export PATH={REMOTE_PY.rsplit('/',1)[0]}:$PATH TMPDIR=/tmp {hf}{threads}; "
               f"cd {REMOTE_REPO} && mkdir -p reproduction/data && "
               f"{REMOTE_PY} -m reproduction.cluster.train_gpu --vf {job.vf} --variant {job.variant} "
               f"--start {job.inst} --end {job.inst + 1} --experiments {exp}{eta} 2>&1 | tee -a {log}")
        _i, o, e = c.exec_command(cmd, timeout=3600)
        out = o.read().decode(errors="replace") + e.read().decode(errors="replace")
        ok = f"INSTANCE_DONE {job.vf} {job.variant} {job.inst}" in out
        # keep only the PARTIAL_* lines centrally
        partials = "\n".join(l for l in out.splitlines() if l.startswith("PARTIAL_") or l.startswith("INSTANCE_DONE"))
        with open(self.outdir / f"gpu_{job.vf}_{job.variant}.log", "a", encoding="utf-8") as f:
            f.write(partials + "\n")
        return ok, out

    def _poll_gpu(self, c, st: MachineState):
        try:
            _i, o, _e = c.exec_command(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
                "--format=csv,noheader,nounits", timeout=15)
            util, used, total, temp = o.read().decode().strip().split(",")[:4]
            st.gpu_util, st.gpu_mem_used, st.gpu_mem_total, st.gpu_temp = (
                float(util), float(used), float(total), float(temp))
            st.last_beat = time.time()
        except Exception:
            pass

    # --- worker: one thread per (machine, slot); per_gpu slots share the GPU --- #
    def _worker(self, m, slot: int):
        st = self.state[m["name"]]
        try:
            c = self._connect(m)
        except Exception as exc:
            st.status = "offline"; st.active[slot] = {"job": f"connect failed: {exc}", "started": 0}
            return
        st.last_beat = time.time()
        while True:
            try:
                job = self.q.get_nowait()
            except Empty:
                with self.lock:
                    st.active.pop(slot, None)
                    if not st.active:
                        st.status = "idle"
                break
            with self.lock:
                st.status = "running"; st.active[slot] = {"job": job.key, "started": time.time()}
            if slot == 0:
                self._poll_gpu(c, st)
            t0 = time.time()
            try:
                ok, _out = self._run_job(c, job)
            except Exception as exc:
                ok = False
                with self.lock:
                    st.active[slot] = {"job": f"{job.key} err:{str(exc)[:40]}", "started": t0}
            dt = time.time() - t0
            with self.lock:
                if ok:
                    st.done += 1; st.secs.append(dt)
                    self.done += 1
                    self.all_secs.append(dt); self.per_vf_secs[job.vf].append(dt)
                    self.done_by_cell[f"{job.vf}/{job.variant}"] = \
                        self.done_by_cell.get(f"{job.vf}/{job.variant}", 0) + 1
                else:
                    st.failed += 1; job.attempts += 1
                    if job.attempts < 3:
                        self.q.put(job)
                    else:
                        self.failed_final += 1
            if slot == 0:
                self._poll_gpu(c, st)
        try:
            c.close()
        except Exception:
            pass

    # --- status writer ----------------------------------------------------- #
    def _status_snapshot(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            remaining = self.jobs_total - self.done - self.failed_final
            # adaptive per-VF median time, fall back to weighted prior
            def vf_time(vf):
                s = self.per_vf_secs[vf]
                if s:
                    return sorted(s)[len(s) // 2]
                base = sorted(self.all_secs)[len(self.all_secs) // 2] if self.all_secs else 180.0
                return base * _VF_WEIGHT[vf]
            # total concurrent slots across the fleet = machines x per_gpu
            active = max(1, sum(st.slots for st in self.state.values() if st.status != "offline"))
            # remaining wall time = (sum of remaining per-VF work) / fleet parallelism
            rem_by_vf = {}
            for vf in GPU_VFS:
                # remaining instances of this vf across variants
                done_vf = sum(v for k, v in self.done_by_cell.items() if k.startswith(vf + "/"))
                total_vf = self.jobs_total // len(GPU_VFS)
                rem_vf = max(0, total_vf - done_vf)
                rem_by_vf[vf] = rem_vf * vf_time(vf)
            rem_secs_total = sum(rem_by_vf.values()) / active
            throughput = self.done / elapsed * 60 if elapsed > 0 else 0.0
            return {
                "elapsed_s": elapsed,
                "jobs_total": self.jobs_total, "done": self.done,
                "failed_final": self.failed_final, "remaining": remaining,
                "pct": 100.0 * self.done / self.jobs_total if self.jobs_total else 0,
                "throughput_per_min": throughput,
                "eta_s": rem_secs_total,
                "eta_by_vf_s": {vf: rem_by_vf[vf] / active for vf in GPU_VFS},
                "vf_median_s": {vf: vf_time(vf) for vf in GPU_VFS},
                "done_by_cell": dict(self.done_by_cell),
                "gpu_hours": sum(sum(st.secs) for st in self.state.values()) / 3600,
                "machines": [asdict(st) for st in self.state.values()],
            }

    def _status_loop(self):
        while self.status_running:
            snap = self._status_snapshot()
            tmp = self.outdir / "fleet_status.json.tmp"
            tmp.write_text(json.dumps(snap, indent=2))
            tmp.replace(self.outdir / "fleet_status.json")
            time.sleep(5)

    def run(self):
        for m in self.machines:
            self.state[m["name"]].slots = self.per_gpu
        if self.sync_code:
            print(f"syncing code to {len(self.machines)} machines via SFTP …")
            for m in self.machines:
                try:
                    self._sync_code(m); print(f"  synced {m['name']}")
                except Exception as exc:
                    print(f"  WARN sync failed {m['name']}: {exc}")
        sw = threading.Thread(target=self._status_loop, daemon=True); sw.start()
        workers = [threading.Thread(target=self._worker, args=(m, slot), daemon=True)
                   for m in self.machines for slot in range(self.per_gpu)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        self.status_running = False
        snap = self._status_snapshot()
        (self.outdir / "fleet_status.json").write_text(json.dumps(snap, indent=2))
        print(f"DONE  {self.done}/{self.jobs_total} ok, {self.failed_final} failed, "
              f"{snap['gpu_hours']:.2f} GPU-hours, {snap['elapsed_s']/60:.1f} min wall")


def parse_machines(path: str):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        host, port, user, pw = line.split(":", 3)
        out.append({"name": f"{host}:{port}", "host": host, "port": int(port),
                    "user": user, "password": pw})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", required=True, help="file: host:port:user:password per line")
    ap.add_argument("--variants", nargs="+", default=["v522_merged", "v560_improved"])
    ap.add_argument("--experiments", nargs="+", default=["table1", "eta"])
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--outdir", default="reproduction/fleet/out")
    ap.add_argument("--per-gpu", type=int, default=6,
                    help="concurrent instances per GPU (6 saturates a 3080 Ti on these small models)")
    ap.add_argument("--eta-budgets", nargs="+", type=int, default=None,
                    help="restrict eta ablation budgets, e.g. 10000 for the reduced Fig4-only run")
    ap.add_argument("--cores", type=int, default=48, help="CPU cores per box (sizes BLAS threads/process)")
    ap.add_argument("--no-hf-mirror", action="store_true")
    ap.add_argument("--no-sync-code", action="store_true", help="skip SFTP code push (clones already current)")
    a = ap.parse_args()
    machines = parse_machines(a.machines)
    njobs = len(a.variants) * len(GPU_VFS) * a.instances
    print(f"fleet: {len(machines)} machines x {a.per_gpu}/gpu = {len(machines)*a.per_gpu} slots, "
          f"{njobs} jobs, experiments={a.experiments}, eta_budgets={a.eta_budgets or 'all'}")
    Fleet(machines, a.variants, a.experiments, a.instances, a.outdir,
          hf_mirror=not a.no_hf_mirror, per_gpu=a.per_gpu,
          sync_code=not a.no_sync_code, eta_budgets=a.eta_budgets, cores=a.cores).run()


if __name__ == "__main__":
    main()
