#!/usr/bin/env python3
"""Compara la clasificación en línea contra referencias offline en TODOS los CSV
de formas de onda de una carpeta (y sus subcarpetas), y deja un resumen.

Para cada archivo corre `examples/sonda_contra_referencia.rs`, que compara el
agrupador en línea con dos referencias offline (PCA de las formas de onda +
HDBSCAN, y HDBSCAN sobre el encaje). Nada se guarda a mano: el encaje sale de

  1. --modelo BASE           un modelo ya entrenado, el mismo para todos; o
  2. --usar-encajes          un `encaje.csv` (o `<nombre>_encaje.csv`) junto al
                             archivo, si existe (no necesita GPU); o
  3. por defecto             entrena UMAP con las primeras --entrenar N formas de
                             onda de cada archivo (necesita GPU).

Uso (desde la raíz de umap-mcqueen):

  python3 scripts/compara_carpeta.py datos/                 # entrena por archivo
  python3 scripts/compara_carpeta.py datos/ --modelo modelos/s1
  python3 scripts/compara_carpeta.py datos/ --usar-encajes
  python3 scripts/compara_carpeta.py datos/ --salida resultados/ --entrenar 3000

Deja en --salida (por defecto `resultados_comparacion/`):
  <ruta del csv>.log    la salida completa de la sonda
  <ruta del csv>.json   sus medidas
  resumen.csv           una fila por archivo
y muestra el resumen al final.

Calidad sin verdad de campo: si junto a un archivo hay un `STime.csv` (el tiempo
de cada espiga, en muestras, una fila por espiga y en el mismo orden), la sonda
mide además, para cada método, cuántas unidades son buenas por sí mismas:
violaciones del periodo refractario, contaminación, presencia, SNR y
aislamiento (ver `src/calidad.rs`). Ningún método hace de juez.
Opciones: --tiempos NOMBRE (por defecto STime.csv), --fs (30000), --refractario-ms (2).

Qué CSV se toman: cualquier `*.csv` con al menos --min-columnas columnas (por
defecto 8) y --min-filas filas (por defecto 1000). Así se saltan los encajes
(2 columnas), los de etiquetas y los archivos pequeños. Sólo usa la biblioteca
estándar de Python.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

EJEMPLO = "sonda_contra_referencia"


def columnas_y_filas(ruta, min_filas):
    """(columnas de la primera fila, filas hasta min_filas) sin leer todo."""
    with open(ruta, "r", errors="replace") as f:
        primera = f.readline()
        if not primera.strip():
            return 0, 0
        cols = len(primera.split(","))
        filas = 1
        for _ in f:
            filas += 1
            if filas >= min_filas:
                break
    return cols, filas


def es_numerica(ruta):
    with open(ruta, "r", errors="replace") as f:
        primera = f.readline()
    try:
        [float(x) for x in primera.strip().split(",")]
        return True
    except ValueError:
        return False


def encaje_vecino(ruta):
    for cand in (ruta.with_name(ruta.stem + "_encaje.csv"), ruta.with_name("encaje.csv")):
        if cand.exists() and cand != ruta:
            return cand
    return None


def compila(raiz):
    print(f"Compilando el ejemplo {EJEMPLO} (release)…", flush=True)
    r = subprocess.run(["cargo", "build", "--release", "--example", EJEMPLO], cwd=raiz)
    if r.returncode != 0:
        sys.exit("No compiló el ejemplo; revisa la salida de cargo.")
    bin_ = raiz / "target" / "release" / "examples" / EJEMPLO
    if not bin_.exists():
        bin_ = bin_.with_suffix(".exe")
    return bin_


def valido(x):
    """Un número que se puede comparar: ni None ni NaN.

    La sonda escribe NaN cuando no hay con qué medir (por ejemplo, un tramo en
    el que la referencia lo marcó todo como ruido), y JSON lo trae como null.
    """
    return isinstance(x, (int, float)) and x == x


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if valido(x) else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados_comparacion"))
    ap.add_argument("--modelo", help="ruta base de un modelo entrenado, para todos los archivos")
    ap.add_argument("--usar-encajes", action="store_true",
                    help="usar encaje.csv / <nombre>_encaje.csv junto a cada archivo si existe")
    ap.add_argument("--entrenar", type=int, default=2000,
                    help="formas de onda para entrenar UMAP por archivo (si no hay modelo ni encaje)")
    ap.add_argument("--min-columnas", type=int, default=8)
    ap.add_argument("--min-filas", type=int, default=1000)
    ap.add_argument("--tramos", type=int, default=5)
    ap.add_argument("--lote", type=int, default=200)
    ap.add_argument("--tiempos", default="STime.csv",
                    help="nombre del archivo de tiempos junto a cada Waveforms (vacío para no usarlo)")
    ap.add_argument("--fs", type=float, default=30000.0, help="frecuencia de muestreo de los tiempos (Hz)")
    ap.add_argument("--refractario-ms", type=float, default=2.0)
    ap.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="raíz del repositorio umap-mcqueen")
    a = ap.parse_args()

    carpeta = a.carpeta.resolve()
    salida = a.salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)

    candidatos = sorted(p for p in carpeta.rglob("*.csv") if salida not in p.parents)
    archivos, saltados = [], []
    for p in candidatos:
        if a.tiempos and p.name == a.tiempos:
            saltados.append((p, "archivo de tiempos"))
            continue
        cols, filas = columnas_y_filas(p, a.min_filas)
        if cols < a.min_columnas:
            saltados.append((p, f"{cols} columnas"))
        elif filas < a.min_filas:
            saltados.append((p, f"{filas} filas"))
        elif not es_numerica(p):
            saltados.append((p, "no es numérico (¿cabecera?)"))
        else:
            archivos.append(p)
    print(f"{len(archivos)} archivos de formas de onda en {carpeta} · {len(saltados)} CSV saltados")
    for p, motivo in saltados:
        print(f"   saltado: {p.relative_to(carpeta)} ({motivo})")
    if not archivos:
        sys.exit("No hay nada que comparar.")

    binario = compila(a.raiz)
    filas = []
    for i, p in enumerate(archivos, 1):
        rel = p.relative_to(carpeta)
        base = salida / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        log, js = base.with_suffix(".log"), base.with_suffix(".json")
        args = [str(binario), "--formas", str(p), "--lote", str(a.lote), "--tramos", str(a.tramos),
                "--resumen-json", str(js)]
        vecino = encaje_vecino(p) if a.usar_encajes else None
        if a.modelo:
            args += ["--modelo", a.modelo]
            origen = "modelo"
        elif vecino:
            args += ["--puntos", str(vecino)]
            origen = f"encaje: {vecino.name}"
        else:
            args += ["--entrenar", str(a.entrenar)]
            origen = f"entrena {a.entrenar}"
        tiempos = p.with_name(a.tiempos) if a.tiempos else None
        if tiempos and tiempos.exists():
            args += ["--tiempos", str(tiempos), "--fs", str(a.fs), "--refractario-ms", str(a.refractario_ms)]
            origen += " · con tiempos"
        print(f"[{i}/{len(archivos)}] {rel} ({origen})…", end=" ", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=a.raiz)
        dt = time.time() - t0
        fila = {"archivo": str(rel), "origen": origen, "segundos": round(dt, 1), "error": ""}
        if r.returncode != 0 or not js.exists():
            ultima = log.read_text(errors="replace").strip().splitlines()[-1:] or ["?"]
            fila["error"] = ultima[0][:200]
            print(f"ERROR ({fila['error']})")
        else:
            d = json.loads(js.read_text())
            fila.update(espigas=d["espigas"], ids=d["ids_emitidos"], uniones=d["uniones"])
            for clave, c in zip(("pca", "encaje"), d["comparaciones"]):
                fila[f"ari_{clave}"] = c["ari_global"]
                tramos = [x for x in c["ari_por_tramo"] if valido(x)]
                fila[f"ari_{clave}_min_tramo"] = min(tramos) if tramos else None
                fila[f"unidades_ref_{clave}"] = c["unidades_referencia"]
                fila[f"fusiones_{clave}"] = len(c["fusiones"])
                fila[f"divisiones_{clave}"] = len(c["divisiones"])
                fila[f"ruido_ref_{clave}"] = c["ruido_referencia"]
                fila[f"ari_{clave}_tolerante"] = c.get("ari_tolerante")
                fila[f"pureza_{clave}"] = c.get("pureza")
            for clave, c in (d.get("calidad") or {}).items():
                fila[f"buenas_{clave}"] = c["n_buenas"]
                fila[f"unidades_{clave}"] = len(c["unidades"])
                fila[f"espigas_buenas_{clave}"] = c["fraccion_espigas_buenas"]
                fila["tiempo_muerto_ms"] = c["tiempo_muerto_ms"]
                fila["contaminacion_medible"] = c["contaminacion_medible"]
            fila["unidades_en_linea"] = d["comparaciones"][0]["unidades_en_linea"]
            fila["ruido_en_linea"] = d["comparaciones"][0]["ruido_en_linea"]
            print(f"ARI {fmt(fila['ari_pca'])} (PCA) · {fmt(fila['ari_encaje'])} (encaje) · {dt:.0f} s")
        filas.append(fila)

    # resumen.csv
    campos = ["archivo", "origen", "espigas", "unidades_en_linea", "unidades_ref_pca", "unidades_ref_encaje",
              "ari_pca", "ari_pca_min_tramo", "ari_encaje", "ari_encaje_min_tramo",
              "ari_pca_tolerante", "pureza_pca", "ari_encaje_tolerante", "pureza_encaje",
              "fusiones_pca", "divisiones_pca", "fusiones_encaje", "divisiones_encaje",
              "ruido_en_linea", "ruido_ref_pca", "ruido_ref_encaje",
              "buenas_en_linea", "buenas_referencia_pca", "buenas_referencia_encaje",
              "espigas_buenas_en_linea", "espigas_buenas_referencia_pca", "espigas_buenas_referencia_encaje",
              "tiempo_muerto_ms", "contaminacion_medible",
              "ids", "uniones", "segundos", "error"]
    with open(salida / "resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow(fila)

    # Resumen en pantalla
    ok = [f for f in filas if not f["error"]]
    print("\n" + "=" * 100)
    print(f"RESUMEN · {len(ok)} de {len(filas)} archivos comparados · detalle en {salida / 'resumen.csv'}")
    print("=" * 100)
    cab = f"{'archivo':<40} {'espigas':>8} {'unid':>5} {'ref':>4} {'ARI PCA':>8} {'mín':>6} {'ARI enc':>8} {'fus':>4} {'div':>4} {'ruido':>6}"
    print(cab)
    for f in filas:
        nombre = f["archivo"] if len(f["archivo"]) <= 40 else "…" + f["archivo"][-39:]
        if f["error"]:
            print(f"{nombre:<40} ERROR: {f['error'][:55]}")
            continue
        print(f"{nombre:<40} {f['espigas']:>8} {f['unidades_en_linea']:>5} {f['unidades_ref_pca']:>4} "
              f"{fmt(f['ari_pca']):>8} {fmt(f['ari_pca_min_tramo']):>6} {fmt(f['ari_encaje']):>8} "
              f"{f['fusiones_pca']:>4} {f['divisiones_pca']:>4} {fmt(100 * f['ruido_en_linea'] if valido(f['ruido_en_linea']) else None, 1):>5}%")
    if ok:
        aris = [f["ari_pca"] for f in ok if valido(f["ari_pca"])]
        if aris:
            print(f"\nARI contra PCA offline: mediana {statistics.median(aris):.3f} · "
                  f"mínimo {min(aris):.3f} · máximo {max(aris):.3f}")
        tol = [f["ari_pca_tolerante"] for f in ok if valido(f.get("ari_pca_tolerante"))]
        pur = [f["pureza_pca"] for f in ok if valido(f.get("pureza_pca"))]
        if tol:
            print(f"ARI tolerando divisiones: mediana {statistics.median(tol):.3f} · "
                  f"pureza mediana {statistics.median(pur):.3f}" if pur else "")
        dudosos = [f for f in ok if not valido(f["ari_pca"]) or f["ari_pca"] < 0.9
                   or f["fusiones_pca"] or f["divisiones_pca"]
                   or f["unidades_en_linea"] != f["unidades_ref_pca"]]
        if dudosos:
            print(f"Para revisar ({len(dudosos)}): ARI < 0.9, fusiones, divisiones o distinto número de unidades:")
            for f in dudosos:
                print(f"   {f['archivo']}  (log: {(salida / f['archivo']).with_suffix('.log')})")
        else:
            print("Ningún archivo con ARI < 0.9, fusiones, divisiones o distinto número de unidades.")
    con_cal = [f for f in ok if "buenas_en_linea" in f]
    if con_cal:
        print(f"\nCALIDAD sin verdad de campo ({len(con_cal)} archivos con tiempos): unidades buenas por método")
        for clave, nombre in (("en_linea", "en línea"), ("referencia_pca", "PCA offline"),
                              ("referencia_encaje", "HDBSCAN sobre el encaje")):
            buenas = sum(f.get(f"buenas_{clave}", 0) for f in con_cal)
            total = sum(f.get(f"unidades_{clave}", 0) for f in con_cal)
            esp = [f[f"espigas_buenas_{clave}"] for f in con_cal if valido(f.get(f"espigas_buenas_{clave}"))]
            med = f"{100 * statistics.median(esp):.1f} %" if esp else "—"
            print(f"   {nombre:<26} {buenas:>5} buenas de {total:>5} unidades · "
                  f"mediana de espigas en unidades buenas por archivo: {med}")
        sin_ventana = [f for f in con_cal if f.get("contaminacion_medible") is False]
        if sin_ventana:
            print(f"   ({len(sin_ventana)} archivos sin ventana entre el tiempo muerto del detector y el "
                  f"refractario: ahí no se calcula la contaminación de Hill, que es informativa; el "
                  f"criterio de unidad buena usa la fracción de intervalos cortos.)")
    print("\nColumnas: unid = unidades en línea (≥0.5 %) · ref = unidades de la referencia PCA · "
          "mín = peor tramo de la sesión · fus/div = fusiones/divisiones contra PCA · ruido = en línea.")
    print("Recuerda: las referencias offline son otro método, no verdad de campo.")


if __name__ == "__main__":
    main()
