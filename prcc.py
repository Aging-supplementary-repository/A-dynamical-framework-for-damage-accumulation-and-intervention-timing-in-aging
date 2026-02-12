import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, qmc
from multiprocessing import Pool, cpu_count
import itertools, os, math, random
from functools import partial

SEED = 2026
np.random.seed(SEED)
random.seed(SEED)

OUTDIR = "outputs_prcc"
os.makedirs(OUTDIR, exist_ok=True)

N_LHS = 3000 
M_STOCH = 50 
N_CORES = max(1, cpu_count() - 1)

param_defs = [
    ("alpha",   "linear", 0.01, 0.10),
    ("beta",    "linear", 0.002, 0.05),
    ("R_Amax",  "linear", 0.05, 0.60),
    ("R_B",     "linear", 0.00, 0.10),
    ("mu",      "log",    1e-5, 5e-3),
    ("b",       "linear", 0.01, 0.20),
    ("d",       "linear", 0.01, 0.20),
    ("N_comp",  "categorical", [1, 2, 5, 10])
]

P = len(param_defs)

def rhs_det(y, alpha, beta, R_A, R_B):
    A, B = y
    return np.array([alpha - R_A * A, beta - R_B * B])

def integrate_det(params, T=500.0, dt=0.5):
    alpha, beta, R_Amax, R_B = params[0], params[1], params[2], params[3]
    R_A = R_Amax
    y = np.array([1.0, 1.0])
    n = int(T / dt)
    D_vals = np.zeros(n)
    for i in range(n):
        k1 = rhs_det(y, alpha, beta, R_A, R_B)
        k2 = rhs_det(y + dt * k1 / 2, alpha, beta, R_A, R_B)
        k3 = rhs_det(y + dt * k2 / 2, alpha, beta, R_A, R_B)
        k4 = rhs_det(y + dt * k3, alpha, beta, R_A, R_B)
        y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        D_vals[i] = y.sum()
    tail = int(0.2 * n)
    slope = (D_vals[-1] - D_vals[-tail]) / (tail * dt)
    return slope

def gillespie_clone(params, Tmax=300.0, seed=None):
    if seed is not None:
        random.seed(seed)
    mu, b, d, N_comp = params[4], params[5], params[6], int(params[7])
    t = 0.0
    M = 0
    Mmax = 0
    while t < Tmax and M < 1e6:
        rate_birth = b * M
        rate_death = d * M
        rate_seed = mu * N_comp
        total = rate_birth + rate_death + rate_seed
        if total <= 0:
            break

        t += random.expovariate(total)
        r = random.random() * total
        if r < rate_birth:
            M += 1
        elif r < rate_birth + rate_death:
            M = max(0, M - 1)
        else:
            M += 1
        Mmax = max(Mmax, M)
    return Mmax

def run_stochastic(X, sample_id):
    params = X[sample_id]
    Mmax_vals = []
    for k in range(M_STOCH):
        seed = SEED + 100000 * sample_id + k
        Mmax_vals.append(gillespie_clone(params, seed=seed))
    return np.mean(Mmax_vals)

def prcc(X, Y):
    Xr = np.array([rankdata(X[:, i]) for i in range(X.shape[1])]).T
    Yr = rankdata(Y)
    prcc_vals = []
    for i in range(X.shape[1]):
        idx = [j for j in range(X.shape[1]) if j != i]
        Xi = Xr[:, i]
        Xo = Xr[:, idx]
        beta_i, *_ = np.linalg.lstsq(Xo, Xi, rcond=None)
        beta_y, *_ = np.linalg.lstsq(Xo, Yr, rcond=None)
        ri = Xi - Xo @ beta_i
        ry = Yr - Xo @ beta_y
        prcc_vals.append(np.corrcoef(ri, ry)[0,1])
    return np.array(prcc_vals)

if __name__ == '__main__':
    sampler = qmc.LatinHypercube(d=P, seed=SEED)
    U = sampler.random(N_LHS)

    X = np.zeros((N_LHS, P))
    param_names = []

    for i, (name, ptype, *vals) in enumerate(param_defs):
        param_names.append(name)
        u = U[:, i]
        if ptype == "linear":
            lo, hi = vals
            X[:, i] = lo + u * (hi - lo)
        elif ptype == "log":
            lo, hi = vals
            X[:, i] = 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
        elif ptype == "categorical":
            choices = vals[0]
            bins = np.linspace(0, 1, len(choices) + 1)
            idx = np.digitize(u, bins) - 1
            idx[idx == len(choices)] = len(choices) - 1
            X[:, i] = np.array(choices)[idx]


    np.savetxt(os.path.join(OUTDIR, "lhs_samples.csv"), X, delimiter=",",
               header=",".join(param_names), comments="")

    print("Running deterministic ODEs...")
    slope_D = np.array([integrate_det(X[i]) for i in range(N_LHS)])

    print("Running stochastic simulations...")
    run_stoch_partial = partial(run_stochastic, X)
    with Pool(N_CORES) as pool:
        Mmax_mean = pool.map(run_stoch_partial, range(N_LHS))

    Mmax_mean = np.array(Mmax_mean)

    prcc_slope = prcc(X, slope_D)
    prcc_Mmax = prcc(X, Mmax_mean)

    np.savetxt(os.path.join(OUTDIR, "prcc_slopeD.csv"),
               np.column_stack([param_names, prcc_slope]),
               fmt="%s", delimiter=",")

    order = np.argsort(np.abs(prcc_slope))[::-1]
    plt.figure(figsize=(6,4))
    plt.barh(np.array(param_names)[order], prcc_slope[order])
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("PRCC with asymptotic damage slope")
    plt.title("Global sensitivity (LHS–PRCC)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "tornado_prcc_slopeD.png"), dpi=300)
    plt.show()

    print("Pipeline complete. Outputs in:", OUTDIR)

