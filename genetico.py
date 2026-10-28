"""Algoritmo genetico (solo NumPy). Cromosoma: n pesos en [0,1] + 1 gen de umbral (x8)."""
import numpy as np

def fitness(ind, X, y, U, P, S):
    w, tau = ind[:-1], ind[-1] * 8
    def dist(A): return np.sqrt((w * ((A[:, None, :] - P[None]) / S) ** 2).sum(2))
    D = dist(X); ok = ((D.argmin(1) == y) & (D.min(1) <= tau)).mean()
    rej = (dist(U).min(1) > tau).mean() if len(U) else 0.0
    return 0.6 * ok + 0.4 * rej

def evolucionar(X, y, U, P, S, pob=40, gens=30, torneo=3, pm=0.1, sigma=0.15, seed=0):
    rng = np.random.default_rng(seed); ng = P.shape[1] + 1
    poblacion = rng.random((pob, ng)); hist = []
    for _ in range(gens + 1):
        apt = np.array([fitness(i, X, y, U, P, S) for i in poblacion])
        hist.append((float(apt.max()), float(apt.mean())))
        nueva = [poblacion[apt.argmax()].copy()]                       # elitismo
        def sel():
            idx = rng.integers(0, pob, torneo); return poblacion[idx[apt[idx].argmax()]]
        while len(nueva) < pob:
            a, b = sel(), sel()
            hijo = np.where(rng.random(ng) < 0.5, a, b)                # cruce uniforme
            m = rng.random(ng) < pm
            hijo = np.clip(hijo + m * rng.normal(0, sigma, ng), 0, 1)  # mutacion
            nueva.append(hijo)
        poblacion = np.array(nueva)
    return poblacion[0], hist
