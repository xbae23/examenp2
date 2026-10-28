"""Detector de figuras: preprocesado, caracteristicas, grafo, busqueda y clasificacion.
Solo usa NumPy, SciPy, scikit-image y NetworkX (todos disponibles en Pyodide)."""
import json, math
import numpy as np, networkx as nx
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

FN = ["vertices", "solidez", "aspecto", "huecos", "CV radial", "log CV radial", "Hu1", "Hu2", "Hu3", "Hu4"]

def preprocesar(gris):
    """gris: array uint8 (H,W). Devuelve mascara, contorno externo (N,2) y numero de huecos."""
    g = filters.gaussian(gris.astype(float), sigma=1, preserve_range=True)
    m = g < filters.threshold_otsu(g)
    if m.mean() > 0.5: m = ~m                      # figura clara sobre fondo oscuro
    k = morphology.disk(1); m = ndi.binary_opening(ndi.binary_closing(m, k), k)
    lab, n = ndi.label(m)
    if n == 0: return None, None, 0
    m = lab == (1 + np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    bg, nb = ndi.label(~m)
    borde = set(np.unique(np.r_[bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    huecos = sum(1 for k in range(1, nb + 1) if k not in borde and (bg == k).sum() > 30)
    cs = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
    c = max(cs, key=len) - 1
    return m, c, huecos

def _perim(c): return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
def _area(p): x, y = p[:, 1], p[:, 0]; return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _hull(p):
    from scipy.spatial import ConvexHull
    return p[ConvexHull(p).vertices]

def _aspecto(h):
    mejor = None
    for a in np.deg2rad(np.arange(0, 90, 1.0)):
        r = h @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        w, l = np.ptp(r[:, 0]), np.ptp(r[:, 1])
        if mejor is None or w * l < mejor[0]: mejor = (w * l, max(w, l) / max(1e-9, min(w, l)))
    return mejor[1]

def poligono(c):
    """Douglas-Peucker con epsilon = 2% del perimetro. Devuelve vertices (n,2) sin repetir el primero."""
    p = measure.approximate_polygon(c, tolerance=0.02 * _perim(c))
    return p[:-1] if np.allclose(p[0], p[-1]) else p

def caracteristicas(m, c, huecos):
    p = poligono(c); n = len(p); h = _hull(c)
    sol = _area(c) / max(1e-9, _area(h))
    cen = np.array(ndi.center_of_mass(m)); r = np.linalg.norm(c - cen, axis=1)
    cv = float(r.std() / r.mean())
    mu = measure.moments_central(m.astype(float), order=3)
    hu = measure.moments_hu(measure.moments_normalized(mu))[:4]
    hu = [-np.sign(v) * math.log10(max(abs(v), 1e-7)) for v in hu]
    circ = 4 * math.pi * _area(c) / _perim(c) ** 2
    v = [n, sol, _aspecto(h), huecos, cv, math.log10(cv + 1e-4), *hu]
    return np.array(v, float), circ

def grafo(c):
    """Grafo representativo: ciclo de n vertices con angulo interior y aristas con longitud normalizada."""
    p = poligono(c); n = len(p)
    sgn = 1 if _signed(p) > 0 else -1
    L = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1); L = L / L.sum()
    G = nx.cycle_graph(n) if n > 2 else nx.path_graph(n)
    for i in range(n):
        a, b, d = p[i - 1] - p[i], p[(i + 1) % n] - p[i], None
        ang = math.degrees(math.acos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
        cruz = a[0] * b[1] - a[1] * b[0]
        if cruz * sgn < 0: ang = 360 - ang       # vertice concavo
        G.nodes[i]["ang"] = ang
        G.edges[i, (i + 1) % n]["long"] = float(L[i]) if n > 2 else 1.0
    return G

def _signed(p): x, y = p[:, 1], p[:, 0]; return np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))

def secuencia(G, modo="bfs"):
    """Recorrido BFS o DFS desde el nodo 0 (orden de vertices)."""
    f = nx.bfs_tree if modo == "bfs" else nx.dfs_tree
    return list(f(G, 0).nodes)

def _atrib(G):
    n = G.number_of_nodes()
    return (np.array([G.nodes[i]["ang"] for i in range(n)]),
            np.array([G.edges[i, (i + 1) % n]["long"] for i in range(n)]))

def dist_grafo(G1, G2):
    """Minimo sobre desplazamientos ciclicos y ambos sentidos. inf si difiere el numero de vertices."""
    if G1.number_of_nodes() != G2.number_of_nodes(): return math.inf
    a1, l1 = _atrib(G1); a2, l2 = _atrib(G2); n = len(a1); mejor = math.inf
    for sentido in (1, -1):
        if sentido == 1: A, Lg = a2, l2
        else: A, Lg = a2[::-1], np.roll(l2[::-1], -1)  # recorrido inverso: arista i une vertices i-1 e i
        for s in range(n):
            mejor = min(mejor, float(np.abs(a1 - np.roll(A, s)).sum() / 180 + np.abs(l1 - np.roll(Lg, s)).sum()))
    return mejor

class Detector:
    def __init__(self, ruta="plantillas.json"):
        d = json.load(open(ruta)); self.d = d
        self.nombres = d["nombres"]; self.P = np.array(d["P"]); self.S = np.array(d["S"])
        self.w = np.array(d["w"]); self.tau = d["tau"]
        self.GP = []
        for g in d["grafos"]:
            G = nx.cycle_graph(len(g["ang"])) if len(g["ang"]) > 2 else nx.path_graph(len(g["ang"]))
            n = len(g["ang"])
            for i in range(n): G.nodes[i]["ang"] = g["ang"][i]; G.edges[i, (i + 1) % n]["long"] = g["long"][i]
            self.GP.append(G)

    def distancias(self, x):
        return np.sqrt((self.w * ((x - self.P) / self.S) ** 2).sum(1))

    def clasificar(self, gris):
        m, c, hu = preprocesar(gris)
        if m is None or len(c) < 20: return {"ok": False, "msg": "No se encontro ninguna figura"}
        x, circ = caracteristicas(m, c, hu); G = grafo(c)
        D = self.distancias(x); i = int(np.argmin(D)); dmin = float(D[i])
        gd = dist_grafo(G, self.GP[i]); orden = np.argsort(D)[:3]
        reconocida = dmin <= self.tau and self.tau > 0
        conf = max(0.0, 1 - dmin / self.tau) if self.tau > 0 else 0.0
        if reconocida and not math.isfinite(gd): conf *= 0.7     # el grafo no verifica la propuesta
        ang, lon = _atrib(G)
        return {"ok": True, "reconocida": bool(reconocida), "figura": self.nombres[i], "conf": conf,
                "dmin": dmin, "tau": self.tau, "grafo_d": gd if math.isfinite(gd) else None,
                "top3": [(self.nombres[int(k)], float(D[k])) for k in orden],
                "x": x.tolist(), "circ": circ, "ang": ang.round(1).tolist(), "long": lon.round(3).tolist(),
                "bfs": secuencia(G, "bfs"), "dfs": secuencia(G, "dfs"),
                "contorno": (c[::max(1, len(c) // 200)] ).round(1).tolist(), "poly": poligono(c).round(1).tolist()}
