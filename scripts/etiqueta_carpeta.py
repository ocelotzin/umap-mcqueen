#!/usr/bin/env python3
"""Corre umap-mcqueen sobre todos los archivos de formas de onda de una carpeta
(y sus subcarpetas) y deja, junto a cada uno, sus etiquetas en mcqueen_labels.csv.

Para cada archivo se llama a la CLI de umap-mcqueen con --etiquetas: una
etiqueta por forma de onda, una por línea y en el orden del CSV (−1 = ruido),
incluidas las de entrenamiento y las del último lote (ver la ayuda de la CLI).
Esas etiquetas se comparan después con una verdad de campo con
scripts/compara_etiquetas.py (o directamente con --comparar).

Uso (desde la raíz de umap-mcqueen):

  python3 scripts/etiqueta_carpeta.py carpeta/                       # entrena un UMAP por archivo
  python3 scripts/etiqueta_carpeta.py carpeta/ --train 3000 --paralelo 4
  python3 scripts/etiqueta_carpeta.py carpeta/ --modelo modelos/s1   # un modelo ya entrenado, para todos
  python3 scripts/etiqueta_carpeta.py carpeta/ --entrenar-con carpeta/canal_0/spikes.csv
  python3 scripts/etiqueta_carpeta.py carpeta/ --comparar            # y luego compara con labels.csv

De dónde sale el modelo de cada archivo:
  * por defecto: UMAP entrenado con las primeras --train formas de onda del propio
    archivo (2 000), con --semilla (9999);
  * --modelo BASE: un modelo ya entrenado (BASE.bin y BASE.json), el mismo para todos;
  * --entrenar-con CSV: entrena UNA vez con las primeras --train formas de onda de
    ese archivo, guarda el modelo en <carpeta>/modelo_comun/ y lo usa para todos.
  Con --modelo o --entrenar-con, todos los archivos comparten encaje y
  normalización: los que tengan otro número de muestras por forma de onda se saltan.

Qué archivos se procesan: los que se llaman como --patron (por defecto spikes.csv)
en cualquier subcarpeta, con al menos --min-columnas columnas (8), primera fila
numérica y al menos --train filas. Con --patron "*.csv" se toman todos los CSV que
cumplan eso (los de etiquetas y tiempos, de 1 o 2 columnas, se saltan solos).

Junto a cada archivo quedan:
  mcqueen_labels.csv   las etiquetas (--nombre-etiquetas para otro nombre)
  mcqueen.log          la salida completa de umap-mcqueen
Un archivo que ya tiene etiquetas se salta, salvo con --sobrescribir. Al final se
escribe <carpeta>/mcqueen_resumen.csv (una fila por archivo: filas, etiquetas,
unidades, ruido, segundos, error) y se muestra un resumen.

Varios archivos a la vez: --paralelo N (por defecto 1). Si cada archivo entrena su
UMAP, todos comparten la GPU; si aparecen errores de memoria, baja N. Ctrl+C
detiene también los procesos en curso. Respeta CARGO_TARGET_DIR. Solo usa la
biblioteca estándar de Python.
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CLI = "umap-mcqueen"


def primera_fila(ruta):
    with open(ruta, "r", errors="replace") as f:
        return f.readline().strip()


def cuenta_filas(ruta):
    with open(ruta, "rb") as f:
        return sum(1 for linea in f if linea.strip())


def columnas_y_numerica(ruta):
    fila = primera_fila(ruta)
    if not fila:
        return 0, False
    campos = fila.split(",")
    try:
        [float(x) for x in campos]
        return len(campos), True
    except ValueError:
        return len(campos), False


def compila(raiz):
    print("Compilando umap-mcqueen (release)…", flush=True)
    r = subprocess.run(["cargo", "build", "--release", "--bin", CLI], cwd=raiz)
    if r.returncode != 0:
        sys.exit("No compiló; revisa la salida de cargo.")
    destino = Path(os.environ["CARGO_TARGET_DIR"]) if os.environ.get("CARGO_TARGET_DIR") else raiz / "target"
    bin_ = destino / "release" / CLI
    return bin_ if bin_.exists() else bin_.with_suffix(".exe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", type=Path)
    ap.add_argument("--patron", default="spikes.csv", help="nombre (o patrón) de los archivos de formas de onda")
    ap.add_argument("--nombre-etiquetas", default="mcqueen_labels.csv")
    ap.add_argument("--train", type=int, default=2000, help="formas de onda de entrenamiento (2000)")
    ap.add_argument("--semilla", type=int, default=9999)
    ap.add_argument("--lote", type=int, default=200)
    ap.add_argument("--modelo", help="ruta base de un modelo ya entrenado, para todos los archivos")
    ap.add_argument("--entrenar-con", type=Path, help="CSV con el que se entrena UNA vez el modelo para todos")
    ap.add_argument("--min-columnas", type=int, default=8)
    ap.add_argument("--sobrescribir", action="store_true", help="rehacer aunque ya existan las etiquetas")
    ap.add_argument("--paralelo", type=int, default=1, help="archivos a la vez (por defecto 1)")
    ap.add_argument("--comparar", action="store_true",
                    help="al terminar, comparar con la verdad (labels.csv) con scripts/compara_etiquetas.py")
    ap.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="raíz del repositorio umap-mcqueen")
    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(0)
    a = ap.parse_args()
    if a.modelo and a.entrenar_con:
        sys.exit("--modelo y --entrenar-con no se combinan: elige una fuente de modelo.")

    carpeta = a.carpeta.resolve()
    if not carpeta.is_dir():
        sys.exit(f"No existe la carpeta {carpeta}.")

    # Columnas que exige un modelo común (si lo hay).
    cols_modelo = None
    if a.entrenar_con:
        a.entrenar_con = a.entrenar_con.resolve()
        if not a.entrenar_con.exists():
            sys.exit(f"No existe {a.entrenar_con}.")
        cols_modelo, numerica = columnas_y_numerica(a.entrenar_con)
        if not numerica:
            sys.exit(f"{a.entrenar_con} no es numérico (¿tiene cabecera?).")
        if cuenta_filas(a.entrenar_con) < a.train:
            sys.exit(f"{a.entrenar_con} tiene menos de {a.train} filas (--train).")

    archivos, saltados = [], []
    for p in sorted(carpeta.rglob(a.patron)):
        if not p.is_file() or p.name == a.nombre_etiquetas:
            continue
        cols, numerica = columnas_y_numerica(p)
        filas = cuenta_filas(p)
        etiquetas = p.with_name(a.nombre_etiquetas)
        if cols < a.min_columnas:
            saltados.append((p, f"{cols} columnas"))
        elif not numerica:
            saltados.append((p, "no es numérico (¿cabecera?)"))
        elif not (a.modelo or a.entrenar_con) and filas < a.train:
            saltados.append((p, f"{filas} filas, menos que --train {a.train}"))
        elif cols_modelo is not None and cols != cols_modelo:
            saltados.append((p, f"{cols} muestras por forma de onda; el modelo común espera {cols_modelo}"))
        elif etiquetas.exists() and not a.sobrescribir:
            saltados.append((p, f"ya tiene {a.nombre_etiquetas} (usa --sobrescribir)"))
        else:
            archivos.append((p, filas))
    print(f"{len(archivos)} archivos para etiquetar en {carpeta} · {len(saltados)} saltados")
    for p, motivo in saltados:
        print(f"   saltado: {p.relative_to(carpeta)} ({motivo})")
    if not archivos:
        sys.exit(0 if saltados else "No hay archivos que etiquetar.")

    cli = compila(a.raiz)

    if a.entrenar_con:
        base = carpeta / "modelo_comun" / a.entrenar_con.stem
        base.parent.mkdir(parents=True, exist_ok=True)
        log = carpeta / "modelo_comun" / "entrenamiento.log"
        print(f"Entrenando el modelo común con las primeras {a.train} formas de onda de {a.entrenar_con} "
              f"(semilla {a.semilla})… log: {log}", flush=True)
        with open(log, "w") as f:
            r = subprocess.run([str(cli), str(a.entrenar_con), "--modelo", str(base), "--train", str(a.train),
                                "--semilla", str(a.semilla), "--lote", str(a.lote)],
                               stdout=f, stderr=subprocess.STDOUT, cwd=a.raiz)
        if r.returncode != 0 or not base.with_suffix(".bin").exists() or not base.with_suffix(".json").exists():
            sys.exit(f"No se pudo entrenar el modelo común. Revisa {log}.")
        a.modelo = str(base)
        print(f"Modelo común guardado en {base}.bin/.json", flush=True)

    candado = threading.Lock()
    procesos = set()

    def procesa(p, filas):
        rel = str(p.relative_to(carpeta))
        etiquetas = p.with_name(a.nombre_etiquetas)
        log = p.with_name("mcqueen.log")
        args = [str(cli), str(p), "--etiquetas", str(etiquetas), "--lote", str(a.lote)]
        if a.modelo:
            args += ["--modelo", a.modelo, "--cargar", "--train", str(min(a.train, filas))]
        else:
            args += ["--train", str(a.train), "--semilla", str(a.semilla)]
        # Un archivo a medias de una corrida anterior no debe contar como hecho.
        if etiquetas.exists():
            etiquetas.unlink()
        t0 = time.time()
        with open(log, "w") as f:
            proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, cwd=a.raiz)
            with candado:
                procesos.add(proc)
            proc.wait()
            with candado:
                procesos.discard(proc)
        fila = {"archivo": rel, "filas": filas, "segundos": round(time.time() - t0, 1), "error": ""}
        if proc.returncode != 0 or not etiquetas.exists():
            ultima = log.read_text(errors="replace").strip().splitlines()[-1:] or ["?"]
            fila["error"] = ultima[0][:200]
            return fila
        et = [int(x) for x in etiquetas.read_text().split()]
        n_et = len(et)
        fila["etiquetas"] = n_et
        if n_et != filas:
            fila["error"] = f"{n_et} etiquetas para {filas} formas de onda"
        unidades = Counter(e for e in et if e >= 0)
        fila["unidades"] = len(unidades)
        fila["unidades_0.5pct"] = sum(1 for c in unidades.values() if c >= 0.005 * n_et)
        fila["ruido"] = round(sum(1 for e in et if e < 0) / n_et, 4) if n_et else ""
        return fila

    origen = "modelo común" if a.entrenar_con else (f"modelo {a.modelo}" if a.modelo else
                                                     f"UMAP por archivo (--train {a.train})")
    paralelo = max(1, min(a.paralelo, len(archivos)))
    print(f"{origen} · {paralelo} archivo(s) a la vez", flush=True)
    filas_res, hechos, t_ini = [], 0, time.time()
    ex = ThreadPoolExecutor(max_workers=paralelo)
    try:
        futuros = [ex.submit(procesa, p, n) for p, n in archivos]
        for fut in as_completed(futuros):
            fila = fut.result()
            filas_res.append(fila)
            hechos += 1
            resto = (time.time() - t_ini) / hechos * (len(archivos) - hechos)
            estado = (f"ERROR: {fila['error']}" if fila["error"] else
                      f"{fila['etiquetas']} etiquetas · {fila['unidades']} unidades · ruido {100 * fila['ruido']:.1f} %")
            print(f"[{hechos}/{len(archivos)}] {fila['archivo']}: {estado} · {fila['segundos']:.0f} s"
                  f" · quedan ~{resto / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        print("\nInterrumpido: deteniendo los procesos en curso…", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        with candado:
            for proc in procesos:
                proc.terminate()
        sys.exit(130)
    ex.shutdown()

    filas_res.sort(key=lambda f: f["archivo"])
    resumen = carpeta / "mcqueen_resumen.csv"
    campos = ["archivo", "filas", "etiquetas", "unidades", "unidades_0.5pct", "ruido", "segundos", "error"]
    with open(resumen, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas_res)
    ok = [f for f in filas_res if not f["error"]]
    print(f"\nEtiquetados {len(ok)} de {len(filas_res)} archivos ({origen}) · resumen en {resumen}")
    for f in filas_res:
        if f["error"]:
            print(f"   con error: {f['archivo']} — {f['error']} (log: {carpeta / f['archivo']} → mcqueen.log)")

    if a.comparar:
        script = Path(__file__).resolve().parent / "compara_etiquetas.py"
        if not script.exists():
            print(f"\nNo encontré {script}; aplica compara-etiquetas.patch para usar --comparar.")
        else:
            print("\nComparando con la verdad de campo…\n", flush=True)
            subprocess.run([sys.executable, str(script), str(carpeta), "--etiquetas", a.nombre_etiquetas,
                            "--salida", str(carpeta / "comparacion_etiquetas")])


if __name__ == "__main__":
    main()
