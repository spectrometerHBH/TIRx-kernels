"""Numpy GDN value oracle: recurrent ground truth + chunked form (the kernel's
algorithm). Cross-validated against each other (run this file directly). Pure
numpy (no torch) so it can back the nymph value-sim test."""

import numpy as np


def recurrent(q, k, v, g, beta, scale, S0=None):
    """Token-by-token ground truth (FLA naive_recurrent_gated_delta_rule).
    q,k:[T,K] v:[T,V] g,beta:[T]. h:[K,V]. Returns o:[T,V], S:[K,V]."""
    T, K = k.shape
    V = v.shape[1]
    h = np.zeros((K, V), np.float64) if S0 is None else S0.astype(np.float64).copy()
    o = np.zeros((T, V), np.float64)
    for i in range(T):
        bq = q[i] * scale
        bk = k[i]
        bv = v[i].copy()
        h = h * np.exp(g[i])
        bv = bv - (h * bk[:, None]).sum(0)  # v - k@h  (k:[K], h:[K,V])
        bv = bv * beta[i]
        h = h + bk[:, None] * bv[None, :]  # h += k ⊗ (beta*delta)
        o[i] = bq @ h  # q @ h
    return o, h


def chunked(q, k, v, g, beta, scale, S0=None, BT=64):
    """Chunked form (FLA naive_chunk_gated_delta_rule) — the kernel's algorithm."""
    T, K = k.shape
    V = v.shape[1]
    pad = (BT - T % BT) % BT
    if pad:
        q = np.pad(q, ((0, pad), (0, 0)))
        k = np.pad(k, ((0, pad), (0, 0)))
        v = np.pad(v, ((0, pad), (0, 0)))
        g = np.pad(g, (0, pad))
        beta = np.pad(beta, (0, pad))
    q = q.astype(np.float64) * scale
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    g = g.astype(np.float64)
    beta = beta.astype(np.float64)
    v = v * beta[:, None]
    k_beta = k * beta[:, None]
    n = (T + pad) // BT
    q, k, v, k_beta = [x.reshape(n, BT, -1) for x in (q, k, v, k_beta)]
    decay = g.reshape(n, BT).cumsum(-1)  # cumsumlog per chunk
    dexp = np.exp(decay)[..., None]
    o = np.zeros((n, BT, V), np.float64)
    S = np.zeros((K, V), np.float64) if S0 is None else S0.astype(np.float64).copy()
    tri0 = np.triu(np.ones((BT, BT), bool), 0)  # incl diagonal
    tri1 = np.triu(np.ones((BT, BT), bool), 1)  # strict
    for i in range(n):
        d = decay[i]
        Lm = np.tril(np.exp(d[:, None] - d[None, :]))  # T[i,j]=exp(d_i-d_j) i>=j
        attn = -((k_beta[i] @ k[i].T) * Lm)
        attn[tri0] = 0.0  # strictly lower-tri
        for r in range(1, BT):  # WY forward substitution
            attn[r, :r] = attn[r, :r] + (attn[r, :r, None] * attn[:r, :r]).sum(0)
        attn = attn + np.eye(BT)  # A_inv = (I+M)^-1
        nv = attn @ v[i]  # new_v base = A_inv @ (v*beta)
        kcd = attn @ (k_beta[i] * dexp[i])  # k_cumdecay
        aqk = (q[i] @ k[i].T) * Lm  # intra attn (causal incl diag)
        aqk[tri1] = 0.0
        v_prime = kcd @ S
        v_new = nv - v_prime
        o_inter = (q[i] * dexp[i]) @ S
        o[i] = o_inter + aqk @ v_new
        glast = d[-1]
        S = S * np.exp(glast) + (k[i] * np.exp(glast - d)[:, None]).T @ v_new
    return o.reshape(n * BT, V)[:T], S


if __name__ == "__main__":
    # Self cross-validation: the chunked form must reproduce the recurrent
    # ground truth to fp64 round-off across varied (incl. partial-chunk) T.
    rng = np.random.default_rng(0)
    maxerr_o = maxerr_s = 0.0
    for trial in range(8):
        T = int(rng.integers(1, 5)) * 64 - int(rng.integers(0, 40))  # varied, incl partial
        T = max(8, T)
        K = V = 128
        q = rng.standard_normal((T, K)) * 0.3
        k = rng.standard_normal((T, K)) * 0.3
        v = rng.standard_normal((T, V)) * 0.3
        g = -np.exp(rng.standard_normal(T) * 0.5) * 0.3  # negative decay gate
        beta = 1 / (1 + np.exp(-rng.standard_normal(T)))  # sigmoid in (0,1)
        scale = 1 / np.sqrt(K)
        S0 = rng.standard_normal((K, V)) * 0.1 if trial % 2 else None
        o1, s1 = recurrent(q, k, v, g, beta, scale, S0)
        o2, s2 = chunked(q, k, v, g, beta, scale, S0)
        eo = np.abs(o1 - o2).max()
        es = np.abs(s1 - s2).max()
        maxerr_o = max(maxerr_o, eo)
        maxerr_s = max(maxerr_s, es)
        print(f"  T={T:4d} S0={'y' if S0 is not None else 'n'}: max|Δo|={eo:.2e} max|ΔS|={es:.2e}")
    agree = "AGREE" if max(maxerr_o, maxerr_s) < 1e-9 else "MISMATCH"
    print(f"OVERALL: max|Δo|={maxerr_o:.2e}  max|ΔS|={maxerr_s:.2e}  -> {agree}")
