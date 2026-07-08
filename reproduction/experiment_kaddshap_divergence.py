import sys, os, time; sys.path.insert(0, os.path.abspath("."))
import numpy as np
# ---------- (1) mechanism demo: ill-conditioned normal-equations solve ----------
print("=== (1) mechanism: np.linalg.solve on a near-singular normal matrix ===")
rng=np.random.RandomState(0); p=300
Q,_=np.linalg.qr(rng.randn(p,p)); eigs=np.logspace(0,-18,p)
A=(Q*eigs)@Q.T; xt=rng.randn(p); b=A@xt+1e-14*rng.randn(p)
cond=eigs[0]/eigs[-1]
phi_solve=np.linalg.solve(A,b); phi_pinv=np.linalg.pinv(A,rcond=1e-10)@b
print(f"  cond(A)={cond:.1e}  solve:max|phi|={np.max(np.abs(phi_solve)):.1e}  pinv:max|phi|={np.max(np.abs(phi_pinv)):.1e}")

# ---------- (2) real kADDSHAP: capture per-instance MSE + cond at diverging vs stable budget ----------
print("=== (2) real corrgroups60 kADDSHAP: instrument solve_regression ===")
import shapiq.approximator.regression.base as regbase
_orig=regbase.solve_regression; _log={"cond":[], "maxphi":[]}
def _wrapped(X, y, kernel_weights):
    kw=kernel_weights
    WX=kw[:,None]*X; M=X.T@WX
    try: _log["cond"].append(float(np.linalg.cond(M)))
    except Exception: _log["cond"].append(np.inf)
    phi=_orig(X,y,kernel_weights); _log["maxphi"].append(float(np.max(np.abs(phi)))); return phi
regbase.solve_regression=_wrapped

from reproduction.core.harness import prepare_vf, make_game, make_estimator, sfv, TABULAR_VFS
loader,kind=next((l,k) for name,l,k,_ in TABULAR_VFS if name=="corrgroups60")
t0=time.time(); vf=prepare_vf(loader,kind,classifier=True); vf.name="corrgroups60"
print(f"  prepared corrgroups60 d={vf.n} in {time.time()-t0:.0f}s")
N=30; DIVERGE=116; STABLE=221
res={DIVERGE:[], STABLE:[]}
for b_ in (DIVERGE, STABLE):
    for i in range(N):
        _log["cond"].clear(); _log["maxphi"].clear()
        truth=vf.ground_truth(vf.x_test[i]); game=make_game(vf.model,vf.background,vf.x_test[i],is_classifier=vf.is_classifier)
        iv=make_estimator("kADDSHAP",vf.n).approximate(b_,game)
        mse=float(np.mean((sfv(iv,vf.n)-truth)**2))
        res[b_].append((mse, max(_log["cond"]) if _log["cond"] else np.nan, max(_log["maxphi"]) if _log["maxphi"] else np.nan))
    mses=[r[0] for r in res[b_]]; conds=[r[1] for r in res[b_]]
    n_blow=sum(1 for m in mses if m>1e6)
    print(f"  budget m={b_}: median MSE={np.median(mses):.2e}  #blown(>1e6)={n_blow}/{N}  median cond={np.nanmedian(conds):.1e}")
# save per-instance for the notebook
import json
json.dump({str(k):[list(map(float,r)) for r in v] for k,v in res.items()}, open("reproduction/paper_reference/kaddshap_divergence_corrgroups60.json","w"))
print("  saved per-instance data")
