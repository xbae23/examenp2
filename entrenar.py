"""Uso offline (python entrenar.py): genera plantillas.json midiendo las 20 figuras de figuras/,
crea imagenes de entrenamiento/prueba aumentadas (rotacion, escala, desplazamiento, ruido, desenfoque)
y optimiza pesos y umbral con el algoritmo genetico."""
import glob, json, os, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import detector as D, genetico as G

rng = np.random.default_rng(42)
rutas = sorted(glob.glob("figuras/*.png"))
nombres = [os.path.basename(r)[3:-4].replace("_", " ") for r in rutas]
imgs = [np.array(Image.open(r).convert("L")) for r in rutas]

def aumentar(im):
    a = rng.uniform(0, 360); s = rng.uniform(0.6, 1.0); c = 255 / 2
    t = rng.uniform(-40, 40, 2); R = np.array([[math.cos(math.radians(a)), -math.sin(math.radians(a))],
                                              [math.sin(math.radians(a)), math.cos(math.radians(a))]]) / s
    out = ndi.affine_transform(im.astype(float), R, offset=np.array([256, 256]) - R @ (np.array([256, 256]) + t), order=1, cval=255)
    out = ndi.gaussian_filter(out, rng.uniform(0, 1.5)) + rng.normal(0, rng.uniform(0, 12), im.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

def desconocida(k):
    im = Image.new("L", (512, 512), 255); d = ImageDraw.Draw(im); cx = cy = 256; r = 170
    if k == 0: pts = [(cx + r * math.cos(2 * math.pi * i / 7), cy + r * math.sin(2 * math.pi * i / 7)) for i in range(7)]  # heptagono
    elif k == 1: pts = [(120, 400), (400, 380), (300, 110)]                                                          # triangulo irregular
    elif k == 2: pts = [(cx + r * math.cos(2 * math.pi * i / 6) * 1.0, cy + r * math.sin(2 * math.pi * i / 6) * 0.55) for i in range(6)]  # hexagono achatado
    else: pts = [(256, 90), (410, 210), (360, 400), (170, 380), (110, 190)]                                          # pentagono irregular
    d.polygon(pts, fill=0); return np.array(im)

def medir(im):
    m, c, h = D.preprocesar(im); return D.caracteristicas(m, c, h)[0], D.grafo(c)

feat, graf = zip(*[medir(i) for i in imgs])
P = np.array(feat); S = P.std(0) + 1e-6; S = np.maximum(S, 1e-3)

def lote(n_por, aug_unk=15):
    X, y = [], []
    for k, im in enumerate(imgs):
        for _ in range(n_por): X.append(medir(aumentar(im))[0]); y.append(k)
    U = [medir(aumentar(desconocida(j % 4)))[0] for j in range(aug_unk * 4)]
    return np.array(X), np.array(y), np.array(U)

Xtr, ytr, Utr = lote(15); Xte, yte, Ute = lote(30, 15)
w0 = np.ones(P.shape[1] + 1); w0[-1] = 0.5
base = G.fitness(w0, Xte, yte, np.empty((0, 10)), P, S)
best, hist = G.evolucionar(Xtr, ytr, Utr, P, S)
w, tau = best[:-1], best[-1] * 8
def evalua(X, y, U):
    dd = lambda A: np.sqrt((w * ((A[:, None] - P[None]) / S) ** 2).sum(2))
    Dx, Du = dd(X), dd(U)
    return (Dx.argmin(1) == y).mean(), ((Dx.argmin(1) == y) & (Dx.min(1) <= tau)).mean(), (Du.min(1) > tau).mean()
a_nothr, a_thr, rej = evalua(Xte, yte, Ute)
res = {"base_pesos_iguales": float((np.sqrt((((Xte[:, None] - P[None]) / S) ** 2).sum(2)).argmin(1) == yte).mean()),
       "acierto_sin_umbral": float(a_nothr), "acierto_con_umbral": float(a_thr), "rechazo_desconocidas": float(rej),
       "ntrain": len(Xtr), "ntest": len(Xte), "nunk": len(Ute)}
json.dump({"nombres": nombres, "FN": D.FN, "P": P.tolist(), "S": S.tolist(), "w": w.tolist(), "tau": float(tau),
           "grafos": [{"ang": [g.nodes[i]["ang"] for i in range(g.number_of_nodes())],
                       "long": [g.edges[i, (i + 1) % g.number_of_nodes()]["long"] for i in range(g.number_of_nodes())]} for g in graf],
           "vertices_modelo": [int(p[0]) for p in P], "hist": hist, "resultados": res}, open("plantillas.json", "w"))
print(json.dumps(res, indent=1)); print("pesos", dict(zip(D.FN, np.round(w, 2))), "tau", round(tau, 3))
