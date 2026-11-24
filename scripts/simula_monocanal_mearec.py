#!/usr/bin/env python3
"""Simula registros extracelulares con MEArec y los exporta como electrodos MONOCANAL.

Cada registro es una sonda (Neuronexus-32 por defecto) con neuronas biofísicas
(los 13 modelos de capa 5 del Blue Brain Project que trae MEArec) colocadas al
azar alrededor de ella. Cada contacto de la sonda es un electrodo monocanal. Los
parámetros de cada registro se sortean, con una semilla maestra, dentro de rangos
fisiológicos (ver RANGOS): número de neuronas, tasas de disparo, ráfagas, ruido,
deriva y amplitudes. Así cada registro es distinto, pero todos son consistentes
con un registro electrofisiológico, y se pueden repetir exactamente.

La exportación sigue los criterios de scripts/hibrido_a_monocanal.py (rama dev):
ventana de 1 ms antes y 2 ms después del pico, SNR de la plantilla media en ese
canal ≥ --snr-min, realineado de cada unidad con su plantilla media (máximo 1 ms)
y al menos --min-unidades unidades visibles por canal. Los archivos son los mismos
que usa monocanal, así que scripts/etiqueta_erolsc.sh, la CLI de dev y los Rmd
funcionan sin cambios.

Hay que correrlo con un Python que tenga MEArec, NEURON y LFPy (en Windows, desde
WSL; ver el README de la rama). Dos pasos:

  # 1. Una sola vez: biblioteca de plantillas (tarda; se reutiliza)
  python simula_monocanal_mearec.py plantillas --out ~/mearec/plantillas.h5

  # 2. Los registros que se quieran, a partir de la biblioteca
  python simula_monocanal_mearec.py registros --plantillas ~/mearec/plantillas.h5 \\
      --out /mnt/c/.../data/mearec --n-registros 6 --duracion 600 --semilla 1

Salidas de `registros` (por registro k):
  OUT/r<k>/r<k>_canal_<ch>/spikes.csv    una fila por espiga, una columna por muestra, sin cabecera
  OUT/r<k>/r<k>_canal_<ch>/labels.csv    unidad verdadera de cada fila
  OUT/r<k>/r<k>_canal_<ch>/times.csv     spike_frame (realineado), spike_time_s, gt_frame
  OUT/r<k>/r<k>_canal_<ch>/unidades.csv  por unidad: espigas, SNR, desplazamiento, tipo celular, deriva
  OUT/r<k>/parametros.json               los parámetros sorteados y las semillas de MEArec
  OUT/r<k>/resumen.csv                   una fila por canal de la sonda (exportado o por qué no)
  OUT/resumen.csv                        una fila por canal exportado de todos los registros
"""
import argparse
import csv
import json
import os
import time

import numpy as np

# Rangos de los parámetros que se sortean en cada registro. Cada entrada es
# (mínimo, máximo) para un valor uniforme, o una lista de opciones.
RANGOS = {
    "n_exc": (8, 20),                 # neuronas excitatorias alrededor de la sonda
    "n_inh": (3, 8),                  # inhibitorias
    "f_exc": (2.0, 8.0),              # tasa media excitatoria (Hz)
    "f_inh": (8.0, 20.0),             # tasa media inhibitoria (Hz)
    "proceso": ["poisson", "gamma"],  # gamma: más regular, con refractariedad
    "gamma_shape": (2.0, 5.0),
    "min_amp": (30.0, 60.0),          # amplitud mínima de una plantilla (µV)
    "max_amp": (250.0, 500.0),        # amplitud máxima (µV)
    "min_dist": (20.0, 30.0),         # distancia mínima entre somas (µm)
    "ruido": (6.0, 14.0),             # desviación estándar del ruido (µV)
    "ruido_color": [False, True],
    "fraccion_rafagas": (0.2, 0.5),   # fracción de neuronas con ráfagas
    "espigas_por_rafaga": (3, 10),
    "duracion_rafaga": (50.0, 150.0), # ms
    "prob_deriva": 0.7,               # probabilidad de que el registro tenga deriva
    "prob_deriva_lenta": 0.7,         # si hay deriva: lenta (continua) o rápida (saltos)
    "velocidad_deriva": (2.0, 15.0),  # µm/min, deriva lenta
    "periodo_deriva_rapida": (20.0, 60.0),  # s
    "salto_deriva_rapida": (5.0, 20.0),     # µm
}


def uniforme(rng, par, entero=False):
    lo, hi = par
    return int(rng.integers(lo, hi + 1)) if entero else float(rng.uniform(lo, hi))


def sortea_parametros(rng, duracion):
    """Parámetros de un registro, dentro de RANGOS."""
    R = RANGOS
    n_exc, n_inh = uniforme(rng, R["n_exc"], True), uniforme(rng, R["n_inh"], True)
    p = {
        "n_exc": n_exc, "n_inh": n_inh,
        "f_exc": round(uniforme(rng, R["f_exc"]), 2), "f_inh": round(uniforme(rng, R["f_inh"]), 2),
        "proceso": str(rng.choice(R["proceso"])),
        "gamma_shape": round(uniforme(rng, R["gamma_shape"]), 2),
        "min_amp": round(uniforme(rng, R["min_amp"]), 1), "max_amp": round(uniforme(rng, R["max_amp"]), 1),
        "min_dist": round(uniforme(rng, R["min_dist"]), 1),
        "ruido": round(uniforme(rng, R["ruido"]), 2), "ruido_color": bool(rng.choice(R["ruido_color"])),
        "n_rafagas": max(1, int(round(uniforme(rng, R["fraccion_rafagas"]) * (n_exc + n_inh)))),
        "espigas_por_rafaga": uniforme(rng, R["espigas_por_rafaga"], True),
        "duracion_rafaga": round(uniforme(rng, R["duracion_rafaga"]), 1),
        "deriva": bool(rng.random() < R["prob_deriva"]),
        "duracion": float(duracion),
    }
    if p["deriva"]:
        p["deriva_modo"] = "slow" if rng.random() < R["prob_deriva_lenta"] else "fast"
        p["velocidad_deriva"] = round(uniforme(rng, R["velocidad_deriva"]), 2)
        p["periodo_deriva_rapida"] = round(uniforme(rng, R["periodo_deriva_rapida"]), 1)
        p["salto_deriva_rapida"] = round(uniforme(rng, R["salto_deriva_rapida"]), 1)
    p["semillas"] = {k: int(rng.integers(1, 2**31 - 1)) for k in ("spiketrains", "templates", "convolution", "noise")}
    return p


def parametros_mearec(mr, p):
    """Traduce los parámetros sorteados al diccionario de MEArec."""
    q = mr.get_default_recordings_params()
    st, tp, rc = q["spiketrains"], q["templates"], q["recordings"]
    st.update(n_exc=p["n_exc"], n_inh=p["n_inh"], f_exc=p["f_exc"], f_inh=p["f_inh"],
              process=p["proceso"], gamma_shape=p["gamma_shape"], duration=p["duracion"], ref_per=2)
    tp.update(min_amp=p["min_amp"], max_amp=p["max_amp"], min_dist=p["min_dist"])
    #Sin `fs`: el registro sale a la frecuencia de las plantillas. Remuestrear
    #(pedir otra fs) falla en MEArec 1.11 con plantillas con deriva; la
    #frecuencia se fija al generar la biblioteca (`plantillas --fs-khz`).
    rc.update(noise_level=p["ruido"], noise_mode="distance-correlated", noise_color=p["ruido_color"],
              filter=True, filter_cutoff=[300, 6000], bursting=True, n_bursting=p["n_rafagas"],
              n_burst_spikes=p["espigas_por_rafaga"], max_burst_duration=p["duracion_rafaga"],
              shape_mod=True, modulation="electrode", drifting=p["deriva"], chunk_duration=10)
    if p["deriva"]:
        rc.update(drift_mode_speed=p["deriva_modo"], drift_mode_probe="rigid",
                  slow_drift_velocity=p["velocidad_deriva"], fast_drift_period=p["periodo_deriva_rapida"],
                  fast_drift_max_jump=p["salto_deriva_rapida"],
                  fast_drift_min_jump=min(5.0, p["salto_deriva_rapida"]))
    q["seeds"] = dict(p["semillas"])
    return q


def exporta_registro(rg, carpeta, prefijo, a, rng):
    """Corta las espigas de cada canal como hibrido_a_monocanal.py y escribe un monocanal por canal."""
    X = np.asarray(rg.recordings)
    n_ch = len(rg.channel_positions)
    if X.shape[0] == n_ch and X.shape[1] != n_ch:
        X = X.T                                        # (muestras, canales)
    fs = float(rg.info["recordings"]["fs"])
    if fs < 1000:                                      # por si alguna versión la da en kHz
        fs *= 1000.0
    n = X.shape[0]
    pre, post = int(round(a.pre_ms * fs / 1000)), int(round(a.post_ms * fs / 1000))
    ms = int(round(a.max_shift_ms * fs / 1000))
    ar = np.arange(-pre, post)

    #MEArec anota cada tren: mtype (modelo de neurona), cell_type (E/I), bursting y,
    #sólo en los que derivan, drifting=True.
    anot = [getattr(s, "annotations", {}) for s in rg.spiketrains]
    gt = {}
    for u, s in enumerate(rg.spiketrains):
        t = np.sort(np.round(np.asarray(s.times.rescale("s").magnitude) * fs).astype(np.int64))
        t = t[(t >= pre + ms) & (t < n - post - ms)]
        if len(t) >= a.min_spikes:
            gt[u] = t

    canales = list(range(n_ch))
    rng.shuffle(canales)
    resumen, exportados = [], []
    for ch in canales:
        if a.canales_por_registro and len(exportados) >= a.canales_por_registro:
            resumen.append({"canal": ch, "estado": "no sorteado"})
            continue
        trace = X[:, ch].astype(np.float32)
        sigma = float(np.median(np.abs(trace)) / 0.6745)
        wfs, labs, frames, gtf, filas = [], [], [], [], []
        for u, t in gt.items():
            w = trace[t[:, None] + ar[None, :]]
            mean = w.mean(axis=0)
            lo, hi = max(0, pre - ms), min(len(ar), pre + ms + 1)
            shift = lo + int(np.argmax(np.abs(mean[lo:hi]))) - pre
            if shift:
                w = trace[(t + shift)[:, None] + ar[None, :]]
                mean = w.mean(axis=0)
            snr = abs(float(mean[pre])) / sigma if sigma > 0 else 0.0
            if snr < a.snr_min:
                continue
            wfs.append(w); labs.append(np.full(len(t), u)); frames.append(t + shift); gtf.append(t)
            filas.append({"unidad": u, "n_spikes": len(t), "snr": round(snr, 2), "desplazamiento_muestras": shift,
                          "tipo": str(anot[u].get("mtype", "")), "excitatoria": anot[u].get("cell_type") == "E",
                          "rafagas": bool(anot[u].get("bursting", False)),
                          "deriva": bool(anot[u].get("drifting", False))})
        if len(filas) < a.min_unidades:
            resumen.append({"canal": ch, "estado": f"solo {len(filas)} unidades visibles", "n_unidades": len(filas)})
            continue
        W, L = np.vstack(wfs), np.concatenate(labs)
        F, G = np.concatenate(frames), np.concatenate(gtf)
        o = np.argsort(F, kind="stable")
        W, L, F, G = W[o], L[o], F[o], G[o]

        nombre = f"{prefijo}_canal_{ch:02d}"
        d = os.path.join(carpeta, nombre)
        os.makedirs(d, exist_ok=True)
        np.savetxt(os.path.join(d, "spikes.csv"), W, fmt="%.3f", delimiter=",")
        np.savetxt(os.path.join(d, "labels.csv"), L, fmt="%d")
        with open(os.path.join(d, "times.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["spike_frame", "spike_time_s", "gt_frame"])
            w.writerows(zip(F.tolist(), np.round(F / fs, 6).tolist(), G.tolist()))
        with open(os.path.join(d, "unidades.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
        fila = {"canal": ch, "estado": "ok", "carpeta": nombre, "n_unidades": len(filas), "n_spikes": len(L),
                "sigma_uV": round(sigma, 2), "snr_mediana": round(float(np.median([r["snr"] for r in filas])), 2)}
        resumen.append(fila)
        exportados.append(fila)
        print(f"   canal {ch:2d}: {len(filas)} unidades, {len(L)} espigas, SNR mediana {fila['snr_mediana']}", flush=True)

    resumen.sort(key=lambda r: r["canal"])
    with open(os.path.join(carpeta, "resumen.csv"), "w", newline="") as f:
        campos = ["canal", "estado", "carpeta", "n_unidades", "n_spikes", "sigma_uV", "snr_mediana"]
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader(); w.writerows(resumen)
    return exportados


def cmd_plantillas(a):
    import MEArec as mr
    p = mr.get_default_templates_params()
    p.update(probe=a.sonda, n=a.n, drifting=True, drift_steps=a.pasos_deriva, seed=a.semilla, dt=1.0 / a.fs_khz)
    t0 = time.time()
    print(f"Plantillas: sonda {a.sonda}, {a.fs_khz:g} kHz, {a.n} por modelo de neurona, deriva en {a.pasos_deriva} pasos, "
          f"semilla {a.semilla}…", flush=True)
    tg = mr.gen_templates(mr.get_default_cell_models_folder(), params=p, n_jobs=a.n_jobs, verbose=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mr.save_template_generator(tg, filename=a.out)
    print(f"{tg.templates.shape[0]} plantillas en {time.time() - t0:.0f} s → {a.out}")


def cmd_registros(a):
    import MEArec as mr
    os.makedirs(a.out, exist_ok=True)
    todos = []
    for k in range(a.primero, a.primero + a.n_registros):
        # Cada registro tiene su propio generador, derivado de la semilla maestra y de k:
        # el registro k es el mismo aunque se generen más o menos registros.
        rng = np.random.default_rng([a.semilla, k])
        p = sortea_parametros(rng, a.duracion)
        prefijo = f"r{k:02d}"
        carpeta = os.path.join(a.out, prefijo)
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "parametros.json"), "w") as f:
            json.dump({"semilla_maestra": a.semilla, "registro": k, "plantillas": os.path.abspath(a.plantillas),
                       **p}, f, indent=2)
        print(f"[{prefijo}] {p['n_exc']} exc + {p['n_inh']} inh · ruido {p['ruido']} µV · "
              f"deriva {p.get('deriva_modo', 'no') if p['deriva'] else 'no'} · {a.duracion:.0f} s", flush=True)
        t0 = time.time()
        rg = mr.gen_recordings(params=parametros_mearec(mr, p), templates=a.plantillas,
                               n_jobs=a.n_jobs, verbose=False)
        print(f"   simulado en {time.time() - t0:.0f} s", flush=True)
        if a.conservar_h5:
            mr.save_recording_generator(rg, filename=os.path.join(carpeta, f"{prefijo}.h5"))
        exp = exporta_registro(rg, carpeta, prefijo, a, rng)
        for e in exp:
            todos.append({"registro": prefijo, **e})
        print(f"   {len(exp)} canales exportados", flush=True)
        del rg
    with open(os.path.join(a.out, "resumen.csv"), "w", newline="") as f:
        campos = ["registro", "canal", "carpeta", "n_unidades", "n_spikes", "sigma_uV", "snr_mediana"]
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader(); w.writerows(todos)
    print(f"Listo: {len(todos)} electrodos monocanal en {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("plantillas", help="genera la biblioteca de plantillas (una vez)")
    t.add_argument("--out", required=True, help="archivo .h5 de salida")
    t.add_argument("--sonda", default="Neuronexus-32", help="sonda de MEAutility (mearec available-probes)")
    t.add_argument("--n", type=int, default=60, help="posiciones y rotaciones por modelo de neurona (60)")
    t.add_argument("--pasos-deriva", type=int, default=30, help="pasos de cada trayectoria de deriva (30)")
    t.add_argument("--fs-khz", type=float, default=30.0,
                   help="frecuencia de muestreo de las plantillas, y de los registros (30: ventanas de 90 muestras)")
    t.add_argument("--semilla", type=int, default=0)
    t.add_argument("--n-jobs", type=int, default=os.cpu_count())
    t.set_defaults(f=cmd_plantillas)

    r = sub.add_parser("registros", help="simula registros y los exporta como monocanal")
    r.add_argument("--plantillas", required=True, help="biblioteca .h5 de `plantillas`")
    r.add_argument("--out", required=True, help="carpeta de salida")
    r.add_argument("--n-registros", type=int, default=4)
    r.add_argument("--primero", type=int, default=0, help="índice del primer registro (para agregar más)")
    r.add_argument("--duracion", type=float, default=600.0, help="segundos por registro (600)")
    r.add_argument("--semilla", type=int, default=1, help="semilla maestra")
    r.add_argument("--canales-por-registro", type=int, default=8,
                   help="canales al azar exportados por registro, entre los que pasan los filtros (0 = todos)")
    r.add_argument("--snr-min", type=float, default=4.0)
    r.add_argument("--min-unidades", type=int, default=2)
    r.add_argument("--min-spikes", type=int, default=50)
    r.add_argument("--pre-ms", type=float, default=1.0)
    r.add_argument("--post-ms", type=float, default=2.0)
    r.add_argument("--max-shift-ms", type=float, default=1.0)
    r.add_argument("--conservar-h5", action="store_true", help="guardar el registro completo de MEArec (~GB)")
    r.add_argument("--n-jobs", type=int, default=1,
                   help="procesos de MEArec (1). Con más, la convolución en paralelo no es reproducible: la "
                        "misma semilla da formas de onda que difieren hasta ~7 µV en algunas espigas")
    r.set_defaults(f=cmd_registros)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
