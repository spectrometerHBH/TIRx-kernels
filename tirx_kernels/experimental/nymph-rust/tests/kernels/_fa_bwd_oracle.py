"""Reference flash-attention BACKWARD math — the value-sim oracle for the nymph
flash_bwd_sm100 port. Mirrors flash_bwd_sm100.py's recompute-from-(Q,K,V,dO,LSE):

  S   = scale · Q·Kᵀ                       [Sq, Sk]   (per (batch,head))
  P   = exp(S − LSE[:,None])               softmax probs (LSE from fwd)
  dPsum = rowsum(dO ∘ O)                    [Sq]       (the "delta", = preprocess output)
  dP  = dO·Vᵀ                              [Sq, Sk]
  dS  = P ∘ (dP − dPsum[:,None])           [Sq, Sk]   (unscaled in-kernel)
  dV  = Pᵀ·dO                              [Sk, Dv]
  dK  = scale · dSᵀ·Q                      [Sk, D]    (kernel scales dK in epilogue)
  dQ  = scale · dS·K                       [Sq, D]    (kernel scales dQ; TMA reduce-add over n-tiles)

Causal: mask S[i,j] for j > i (+ offset). All math in float32.
"""

import numpy as np


def softmax_lse(s_scaled: np.ndarray) -> np.ndarray:
    """Row LSE = log(sum exp(s)) with max-shift, matching the fwd's stored LSE."""
    m = s_scaled.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(s_scaled - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def fa_bwd(q, k, v, do, o, lse, scale, *, causal=False):
    """One (Sq,D) head. q,k,v,do,o float32 [S,D]; lse [Sq]; returns dq,dk,dv [.,D]."""
    q, k, v, do, o = (x.astype(np.float32) for x in (q, k, v, do, o))
    sq, d = q.shape
    sk = k.shape[0]
    s = scale * (q @ k.T)                                   # [Sq, Sk]
    if causal:
        # token i (row) attends to j <= i + (Sk - Sq)
        ii = np.arange(sq)[:, None]
        jj = np.arange(sk)[None, :]
        s = np.where(jj <= ii + (sk - sq), s, -np.inf)
    p = np.exp(s - lse[:, None])                            # [Sq, Sk]
    p = np.where(np.isfinite(p), p, 0.0)
    dpsum = (do * o).sum(axis=-1)                           # [Sq]  (delta)
    dp = do @ v.T                                           # [Sq, Sk]
    ds = p * (dp - dpsum[:, None])                          # [Sq, Sk]  (unscaled)
    dv = p.T @ do                                           # [Sk, Dv]
    dk = scale * (ds.T @ q)                                 # [Sk, D]
    dq = scale * (ds @ k)                                   # [Sq, D]
    return dq, dk, dv


def fa_bwd_from_qkv(q, k, v, do, scale, *, causal=False):
    """Convenience: recompute O and LSE from a reference forward, then run bwd.
    q,k,v,do float32 [S,D]. Returns (dq,dk,dv, o, lse) so callers can feed o/lse."""
    q, k, v, do = (x.astype(np.float32) for x in (q, k, v, do))
    sq, sk = q.shape[0], k.shape[0]
    s = scale * (q @ k.T)
    if causal:
        ii = np.arange(sq)[:, None]; jj = np.arange(sk)[None, :]
        s = np.where(jj <= ii + (sk - sq), s, -np.inf)
    lse = softmax_lse(s)
    p = np.exp(s - lse[:, None]); p = np.where(np.isfinite(p), p, 0.0)
    o = p @ v
    dq, dk, dv = fa_bwd(q, k, v, do, o, lse, scale, causal=causal)
    return dq, dk, dv, o, lse
