#!/usr/bin/env python3
"""Compara las etiquetas de umap-mcqueen con la verdad de campo, en todas las
subcarpetas de una carpeta. No compila ni corre nada: solo lee etiquetas.

En cada carpeta que tenga los dos archivos (por defecto `mcqueen_labels.csv`, las
etiquetas de umap-mcqueen, y `labels.csv`, la verdad), una etiqueta por espiga y
en el mismo orden, mide:

  * ARI entre ambas, y ARI tolerando divisiones: cada unidad de mcqueen pasa a la
    neurona verdadera con la que comparte más espigas, de modo que una neurona
    partida en varias unidades puras no cuenta como error (una fusión sí).
  * Pureza: fracción de espigas que pertenecen a la neurona mayoritaria de su
    unidad.
  * Por neurona verdadera (con al menos --min-verdad espigas): la unidad de
    mcqueen que comparte más espigas con ella, recall, precisión y exactitud
    = aciertos / (espigas de la neurona + espigas de la unidad − aciertos), como
    en SpikeForest; y la exactitud tolerando divisiones. Recuperada = exactitud
    ≥ --umbral (0.8).

Son las mismas definiciones que usa examples/sonda_contra_referencia.rs.

Convenciones: una etiqueta negativa es ruido (en mcqueen) o «sin etiqueta» (en la
verdad; esas espigas no se evalúan). Si la primera fila no es numérica, se toma
como cabecera; se usa la primera columna.

Uso:

  python3 scripts/compara_etiquetas.py carpeta_hibridos/
  python3 scripts/compara_etiquetas.py carpeta/ --etiquetas mis_labels.csv --verdad labels.csv --salida res/

Deja en --salida (por defecto `comparacion_etiquetas/`):
  resumen_etiquetas.csv   una fila por carpeta
  neuronas.csv            una fila por neurona verdadera evaluada
y muestra el resumen. Solo usa la biblioteca estándar de Python.
"""

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def lee_etiquetas(ruta):
    """Primera columna como enteros; salta una cabecera no numérica."""
    out = []
    with open(ruta, newline="", errors="replace") as f:
        for i, fila in enumerate(csv.reader(f)):
            if not fila or not fila[0].strip():
                continue
            try:
                out.append(int(round(float(fila[0].strip()))))
            except ValueError:
                if i == 0:
                    continue
                raise ValueError(f"{ruta}, fila {i + 1}: «{fila[0]}» no es un número")
    return out


def c2(x):
    return x * (x - 1) / 2


def ari(a, b):
    """Índice de Rand ajustado (1 = misma partición; ~0 = al azar)."""
    n = len(a)
    if n < 2:
        return float("nan")
    t = Counter(zip(a, b))
    sa = sum(c2(x) for x in Counter(a).values())
    sb = sum(c2(x) for x in Counter(b).values())
    s = sum(c2(x) for x in t.values())
    e = sa * sb / c2(n)
    d = 0.5 * (sa + sb) - e
    return 1.0 if d == 0 else (s - e) / d


def compara(mcq, verdad, min_verdad, umbral):
    # Solo las espigas con verdad; el ruido de mcqueen cuenta como una clase más.
    pares = [(m, v) for m, v in zip(mcq, verdad) if v >= 0]
    m_ev = [m for m, _ in pares]
    v_ev = [v for _, v in pares]

    k = Counter((v, m) for m, v in pares if m >= 0)       # (verdad, unidad) → aciertos
    n_ref = Counter(v_ev)
    n_uni = Counter(m for m in m_ev if m >= 0)

    # Neurona mayoritaria de cada unidad (empates: la de etiqueta menor).
    mayor = {}
    for (v, m), x in k.items():
        if m not in mayor or x > mayor[m][1] or (x == mayor[m][1] and v < mayor[m][0]):
            mayor[m] = (v, x)
    tolerante = [m if m < 0 else mayor[m][0] for m in m_ev]
    con_unidad = sum(n_uni.values())
    pureza = (sum(x for _, x in mayor.values()) / con_unidad) if con_unidad else float("nan")

    neuronas = []
    for v, nv in sorted(n_ref.items(), key=lambda t: (-t[1], t[0])):
        if nv < min_verdad:
            continue
        cand = [(m, x) for (vv, m), x in k.items() if vv == v]
        m_mejor, aciertos = max(cand, key=lambda t: (t[1], -t[0])) if cand else (None, 0)
        nu = n_uni.get(m_mejor, 0)
        exact = aciertos / (nv + nu - aciertos) if aciertos else 0.0
        suyas = [m for m, (vv, _) in mayor.items() if vv == v]
        k_tol = sum(k.get((v, m), 0) for m in suyas)
        n_tol = sum(n_uni.get(m, 0) for m in suyas)
        exact_tol = k_tol / (nv + n_tol - k_tol) if k_tol else 0.0
        neuronas.append({
            "neurona": v, "espigas": nv, "unidad_mcqueen": m_mejor if m_mejor is not None else "",
            "recall": aciertos / nv, "precision": aciertos / nu if nu else 0.0,
            "exactitud": exact, "exactitud_tolerante": exact_tol, "unidades_suyas": len(suyas),
            "recuperada": exact >= umbral, "recuperada_tolerante": exact_tol >= umbral,
        })

    ex = [n["exactitud"] for n in neuronas]
    return {
        "espigas": len(mcq),
        "espigas_con_verdad": len(pares),
        "neuronas_verdad": len(n_ref),
        "neuronas_evaluadas": len(neuronas),
        "unidades_mcqueen": len(n_uni),
        "unidades_mcqueen_0.5pct": sum(1 for x in n_uni.values() if x >= 0.005 * len(pares)),
        "ruido_mcqueen": (sum(1 for m in m_ev if m < 0) / len(m_ev)) if m_ev else float("nan"),
        "ari": ari(m_ev, v_ev),
        "ari_tolerante": ari(tolerante, v_ev),
        "pureza": pureza,
        "recuperadas": sum(n["recuperada"] for n in neuronas),
        "recuperadas_tolerante": sum(n["recuperada_tolerante"] for n in neuronas),
        "exactitud_media": statistics.mean(ex) if ex else float("nan"),
        # Mediana como en la sonda (el elemento central de la lista ordenada;
        # con un número par, el mayor de los dos centrales), para que coincidan.
        "exactitud_mediana": sorted(ex)[len(ex) // 2] if ex else float("nan"),
    }, neuronas


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", type=Path)
    ap.add_argument("--etiquetas", default="mcqueen_labels.csv", help="etiquetas de umap-mcqueen (mcqueen_labels.csv)")
    ap.add_argument("--verdad", default="labels.csv", help="etiquetas verdaderas (labels.csv)")
    ap.add_argument("--min-verdad", type=int, default=50, help="neuronas con menos espigas no se evalúan (50)")
    ap.add_argument("--umbral", type=float, default=0.8, help="exactitud para contar una neurona como recuperada (0.8)")
    ap.add_argument("--salida", type=Path, default=Path("comparacion_etiquetas"))
    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(0)
    a = ap.parse_args()

    carpeta = a.carpeta.resolve()
    pares = sorted(p.parent for p in carpeta.rglob(a.etiquetas) if (p.parent / a.verdad).exists())
    sin_verdad = sorted(p.parent for p in carpeta.rglob(a.etiquetas) if not (p.parent / a.verdad).exists())
    if not pares:
        sys.exit(f"No hay carpetas con {a.etiquetas} y {a.verdad} juntos en {carpeta}.")
    print(f"{len(pares)} carpetas con {a.etiquetas} y {a.verdad}" +
          (f" · {len(sin_verdad)} con {a.etiquetas} pero sin {a.verdad} (se saltan)" if sin_verdad else ""))

    a.salida.mkdir(parents=True, exist_ok=True)
    filas, todas_neuronas = [], []
    for c in pares:
        rel = str(c.relative_to(carpeta)) or "."
        fila = {"carpeta": rel, "error": ""}
        try:
            mcq = lee_etiquetas(c / a.etiquetas)
            ver = lee_etiquetas(c / a.verdad)
            if len(mcq) != len(ver):
                raise ValueError(f"{a.etiquetas} tiene {len(mcq)} filas y {a.verdad} {len(ver)}: "
                                 f"tienen que ser las mismas espigas, en el mismo orden")
            medidas, neuronas = compara(mcq, ver, a.min_verdad, a.umbral)
            fila.update(medidas)
            for n in neuronas:
                todas_neuronas.append({"carpeta": rel, **n})
            print(f"  {rel:<28} ARI {fmt(medidas['ari'])} · tolerante {fmt(medidas['ari_tolerante'])} · "
                  f"pureza {fmt(medidas['pureza'])} · recuperadas {medidas['recuperadas']}/{medidas['neuronas_evaluadas']} "
                  f"· unidades {medidas['unidades_mcqueen']}")
        except Exception as e:  # noqa: BLE001 — se informa y se sigue
            fila["error"] = str(e)[:200]
            print(f"  {rel:<28} ERROR: {fila['error']}")
        filas.append(fila)

    campos = ["carpeta", "espigas", "espigas_con_verdad", "neuronas_verdad", "neuronas_evaluadas",
              "unidades_mcqueen", "unidades_mcqueen_0.5pct", "ruido_mcqueen", "ari", "ari_tolerante", "pureza",
              "recuperadas", "recuperadas_tolerante", "exactitud_media", "exactitud_mediana", "error"]
    with open(a.salida / "resumen_etiquetas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    campos_n = ["carpeta", "neurona", "espigas", "unidad_mcqueen", "recall", "precision", "exactitud",
                "exactitud_tolerante", "unidades_suyas", "recuperada", "recuperada_tolerante"]
    with open(a.salida / "neuronas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos_n)
        w.writeheader()
        w.writerows(todas_neuronas)

    ok = [f for f in filas if not f["error"]]
    if ok:
        ev = sum(f["neuronas_evaluadas"] for f in ok)
        rec = sum(f["recuperadas"] for f in ok)
        rec_t = sum(f["recuperadas_tolerante"] for f in ok)
        aris = [f["ari"] for f in ok if f["ari"] == f["ari"]]
        tols = [f["ari_tolerante"] for f in ok if f["ari_tolerante"] == f["ari_tolerante"]]
        purs = [f["pureza"] for f in ok if f["pureza"] == f["pureza"]]
        exs = [n["exactitud"] for n in todas_neuronas]
        print("\n" + "=" * 90)
        print(f"RESUMEN · {len(ok)} de {len(filas)} carpetas · detalle en {a.salida / 'resumen_etiquetas.csv'} "
              f"y {a.salida / 'neuronas.csv'}")
        print("=" * 90)
        print(f"Neuronas evaluadas (≥ {a.min_verdad} espigas): {ev} · recuperadas (exactitud ≥ {a.umbral}): {rec}"
              f" · tolerando divisiones: {rec_t}")
        if aris:
            print(f"ARI por carpeta: mediana {statistics.median(aris):.3f} (mín. {min(aris):.3f}, máx. {max(aris):.3f})"
                  f" · tolerante: mediana {statistics.median(tols):.3f} · pureza mediana {statistics.median(purs):.3f}")
        if exs:
            print(f"Exactitud por neurona: mediana {statistics.median(exs):.3f} · media {statistics.mean(exs):.3f}")
        print("Una neurona puede aparecer en varias carpetas (por ejemplo, en varios canales): se cuenta en cada una.")


if __name__ == "__main__":
    main()
