#!/usr/bin/env python3
"""Resume la rejilla de calibración de EROLSC sobre registros simulados y elige parámetros.

Lee los erolsc_cal_*.csv que deja examples/sonda_de_erolsc.rs en cada canal de
<carpeta>/r*/r*_canal_* (con columna `ventana`) y reporta:
  - las mejores combinaciones (ventana, Td, Tn, M) por ARI medio, con cada
    registro pesando lo mismo (los canales de un registro no son independientes);
  - la mejor combinación de cada ventana;
  - una validación cruzada dejando un registro fuera, dentro de la calibración,
    para ver si la elección es estable.

Sólo biblioteca estándar.   python scripts/analiza_calibracion.py data/mearec
"""
import csv
import glob
import os
import statistics as st
import sys
from collections import Counter, defaultdict

carpeta = sys.argv[1] if len(sys.argv) > 1 else "data/mearec"
ari = defaultdict(dict); ruido = defaultdict(dict); grupos = defaultdict(dict)
for f in glob.glob(os.path.join(carpeta, "r*", "r*_canal_*", "erolsc_cal_*.csv")):
    canal = os.path.basename(os.path.dirname(f))
    with open(f, newline="") as g:
        for r in csv.DictReader(g):
            k = (int(r["ventana"]), float(r["td"]), float(r["tn"]), int(r["m"]))
            ari[k][canal] = float(r["ari"]); ruido[k][canal] = float(r["ruido"]); grupos[k][canal] = int(r["grupos"])
if not ari:
    sys.exit(f"No hay erolsc_cal_*.csv en {carpeta}/r*/r*_canal_*")
canales = sorted({c for v in ari.values() for c in v})
ari = {k: v for k, v in ari.items() if len(v) == len(canales)}
reg = lambda c: c.split("_")[0]
registros = sorted({reg(c) for c in canales})
print(len(canales), "canales,", len(registros), "registros,", len(ari), "combinaciones completas")


def por_registro(k, regs):
    return st.mean(st.mean(ari[k][c] for c in canales if reg(c) == r) for r in regs)


filas = sorted(((por_registro(k, registros), st.mean(ari[k].values()), k) for k in ari), reverse=True)
print("\n(ventana, Td, Tn, M) | ARI media de registros | media de canales | anomalías | grupos (mediana)")
for pr, pc, k in filas[:15]:
    print(k, "| %.3f | %.3f | %.1f%% | %d" % (pr, pc, 100 * st.mean(ruido[k].values()), st.median(grupos[k].values())))
print("\nmejor por ventana:")
for v in sorted({k[0] for k in ari}):
    pr, pc, k = max(x for x in filas if x[2][0] == v)
    print(v, k[1:], "%.3f" % pr)

print("\nvalidación cruzada dejando un registro fuera:")
elegidos, cv = [], []
for r in registros:
    otros = [x for x in registros if x != r]
    k = max(ari, key=lambda k: por_registro(k, otros))
    elegidos.append(k); cv.append(por_registro(k, [r]))
    print(" ", r, "elegido", k, "ARI en el registro fuera %.3f" % cv[-1])
print("  media %.3f; elecciones: %s" % (st.mean(cv), dict(Counter(elegidos))))
