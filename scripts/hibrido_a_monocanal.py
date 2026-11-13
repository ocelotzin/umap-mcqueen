#!/usr/bin/env python3
"""

PORFAVOR REVISAR https://spikeforest.flatironinstitute.org/

Convierte un NWB HIBRIDO (varias unidades con ground truth) en datasets MONOCANAL.

Para cada electrodo:
  - los puntos son los spikes del ground truth (tiempos GT de todas las unidades), leidos en ESE canal;
    asi todos llevan etiqueta (no hay -1 salvo el ruido opcional de --ruido-pct)
  - solo entran las unidades visibles en ese canal (SNR >= --snr-min); las demas quedarian enterradas en ruido
  - cada unidad se realinea con su plantilla media en ese canal (--sin-realinear para desactivar)
  - el canal se omite si quedan menos de --min-unidades unidades

Salidas:
  OUT/canal_<id>/spikes.csv    una fila por spike, una columna por muestra, sin encabezado
  OUT/canal_<id>/labels.csv    id de la unidad GT por fila (-1 = ruido anadido con --ruido-pct)
  OUT/canal_<id>/times.csv     spike_frame (alineado), spike_time_s, gt_frame (tiempo GT original)
  OUT/canal_<id>/unidades.csv  por unidad: n_spikes, snr y desplazamiento aplicado
  OUT/resumen.csv              una fila por canal

Ejemplos:
  python hibrido_a_monocanal.py hibrido.nwb --out salida
  python hibrido_a_monocanal.py hibrido.nwb --canales 3,4,5 --snr-min 5 --ruido-pct 10
"""
import argparse
import os

import numpy as np
import pandas as pd
import spikeinterface.full as si


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nwb", help="ruta al archivo .nwb hibrido")
    p.add_argument("--gt", default=None, help="NWB con las unidades GT (por defecto, el mismo archivo)")
    p.add_argument("--out", default=".", help="carpeta de salida")
    p.add_argument("--canales", default="all", help="'all' o ids separados por coma")
    p.add_argument("--snr-min", type=float, default=4.0,
                   help="una unidad entra en un canal si |pico de su plantilla| / sigma >= este valor (def. 4)")
    p.add_argument("--min-spikes-unidad", type=int, default=50, help="minimo de spikes GT por unidad (def. 50)")
    p.add_argument("--min-unidades", type=int, default=2, help="minimo de unidades visibles para conservar un canal (def. 2)")
    p.add_argument("--pre-ms", type=float, default=1.0, help="ms antes del pico")
    p.add_argument("--post-ms", type=float, default=2.0, help="ms despues del pico")
    p.add_argument("--max-shift-ms", type=float, default=1.0, help="desplazamiento maximo al realinear (ms)")
    p.add_argument("--sin-realinear", action="store_true", help="usar los tiempos GT tal cual")
    p.add_argument("--ruido-pct", type=float, default=0.0,
                   help="anadir ventanas de ruido (etiqueta -1) equivalentes a este %% de los spikes GT (def. 0)")
    p.add_argument("--fmin", type=float, default=300.0, help="pasa-altas (Hz)")
    p.add_argument("--fmax", type=float, default=6000.0, help="pasa-bajas (Hz)")
    p.add_argument("--cmr", action="store_true", help="referencia comun por mediana")
    p.add_argument("--series", default=None, help="nombre/ruta de la ElectricalSeries si hay varias")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def buscar_series(path):
    import h5py
    rutas = []

    def visitar(name, obj):
        if isinstance(obj, h5py.Group):
            t = obj.attrs.get("neurodata_type", "")
            t = t.decode() if isinstance(t, bytes) else str(t)
            if t == "ElectricalSeries":
                rutas.append(name)

    with h5py.File(path, "r") as f:
        f.visititems(visitar)
    return rutas


def cargar_sorting(path, rec, series=None):
    fs = rec.get_sampling_frequency()
    try:
        t0 = rec.get_start_time() if hasattr(rec, "get_start_time") else 0.0
        t0 = 0.0 if t0 is None else float(t0)
    except Exception:
        t0 = 0.0
    intentos = [("lectura directa", {})]
    for ruta in dict.fromkeys(([series] if series else []) + buscar_series(path)):
        intentos.append((f"electrical_series_path='{ruta}'", {"electrical_series_path": ruta}))
    intentos.append(("t_start + sampling_frequency", {"t_start": t0, "sampling_frequency": fs}))
    intentos.append(("t_start", {"t_start": t0}))
    ultimo = None
    for nombre, kw in intentos:
        try:
            s = si.read_nwb_sorting(path, **kw)
            print(f"Unidades GT leidas con: {nombre}")
            return s
        except Exception as e:  # noqa: BLE001
            ultimo = e
    raise SystemExit(f"No se pudieron leer las unidades GT del NWB. Ultimo error: {ultimo}")


def traza_canal(proc, ch, chunk_s=60):
    fs = proc.get_sampling_frequency()
    n = proc.get_num_samples()
    paso = int(chunk_s * fs)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, paso):
        e = min(n, s + paso)
        out[s:e] = proc.get_traces(start_frame=s, end_frame=e, channel_ids=[ch])[:, 0]
    return out


def codigos_unidades(ids):
    """Etiqueta = id original si es entero >= 0 y unico; si no, 0..n-1."""
    try:
        c = [int(u) for u in ids]
        if min(c) >= 0 and len(set(c)) == len(c):
            return c
    except (TypeError, ValueError):
        pass
    return list(range(len(ids)))


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    kw = {"electrical_series_name": a.series} if a.series else {}
    try:
        rec = si.read_nwb_recording(a.nwb, **kw)
    except TypeError:
        kw = {"electrical_series_path": a.series} if a.series else {}
        rec = si.read_nwb_recording(a.nwb, **kw)
    sorting = cargar_sorting(a.gt or a.nwb, rec, a.series)

    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    pre, post = int(a.pre_ms * fs / 1000), int(a.post_ms * fs / 1000)
    ms = int(a.max_shift_ms * fs / 1000)
    ar = np.arange(-pre, post)

    # --- ground truth por unidad ---
    ids = list(sorting.unit_ids)
    cods = codigos_unidades(ids)
    gt = {}
    for uid, cod in zip(ids, cods):
        t = np.sort(np.asarray(sorting.get_unit_spike_train(uid)).astype(np.int64))
        t = t[(t >= pre + ms) & (t < n - post - ms)]
        if len(t) >= a.min_spikes_unidad:
            gt[cod] = t
    if not gt:
        raise SystemExit("Ninguna unidad tiene suficientes spikes GT (revisa --min-spikes-unidad).")
    todos_gt = np.sort(np.concatenate(list(gt.values())))
    print(f"fs = {fs:.0f} Hz | canales = {rec.get_num_channels()} | unidades GT usables = {len(gt)} de {len(ids)} | "
          f"spikes GT = {len(todos_gt)}")

    # --- preprocesamiento ---
    proc = si.bandpass_filter(rec, freq_min=a.fmin, freq_max=a.fmax)
    if a.cmr and rec.get_num_channels() > 1:
        proc = si.common_reference(proc, reference="global", operator="median")
        print("Referencia comun por mediana activada.")

    canales = list(rec.channel_ids)
    if a.canales.lower() != "all":
        pedidos = [c.strip() for c in a.canales.split(",")]
        canales = [c for c in canales if str(c) in pedidos]
        if not canales:
            raise SystemExit(f"Ninguno de los canales pedidos existe: {a.canales}")

    resumen = []
    for i, ch in enumerate(canales, 1):
        trace = traza_canal(proc, ch)
        sigma = float(np.median(np.abs(trace)) / 0.6745)
        if sigma <= 0:
            resumen.append({"canal": ch, "estado": "sigma = 0"})
            continue

        wfs, labs, frames, gt_frames, filas = [], [], [], [], []
        for cod, t in gt.items():
            w = trace[t[:, None] + ar[None, :]]
            mean = w.mean(axis=0)
            shift = 0
            if not a.sin_realinear:
                lo, hi = max(0, pre - ms), min(len(ar), pre + ms + 1)
                shift = lo + int(np.argmax(np.abs(mean[lo:hi]))) - pre
                if shift != 0:
                    w = trace[(t + shift)[:, None] + ar[None, :]]
                    mean = w.mean(axis=0)
            snr = abs(float(mean[pre])) / sigma
            if snr < a.snr_min:
                continue
            wfs.append(w)
            labs.append(np.full(len(t), cod))
            frames.append(t + shift)
            gt_frames.append(t)
            filas.append({"unidad": cod, "n_spikes": len(t), "snr": round(snr, 2), "desplazamiento_muestras": shift})

        if len(filas) < a.min_unidades:
            print(f"[{i}/{len(canales)}] canal {ch}: {len(filas)} unidad(es) visible(s) (< {a.min_unidades}), omitido")
            resumen.append({"canal": ch, "estado": f"solo {len(filas)} unidades visibles", "n_unidades": len(filas)})
            continue

        W = np.vstack(wfs)
        L = np.concatenate(labs)
        F = np.concatenate(frames)
        G = np.concatenate(gt_frames)
        n_gt = len(L)

        n_ruido = 0
        if a.ruido_pct > 0:
            objetivo = int(round(a.ruido_pct / 100 * n_gt))
            cand = rng.integers(pre, n - post, size=objetivo * 4 + 10)
            pos = np.searchsorted(todos_gt, cand)
            izq, der = np.clip(pos - 1, 0, len(todos_gt) - 1), np.clip(pos, 0, len(todos_gt) - 1)
            lejos = np.minimum(np.abs(todos_gt[izq] - cand), np.abs(todos_gt[der] - cand)) > int(0.002 * fs)
            cand = np.unique(cand[lejos])[:objetivo]
            n_ruido = len(cand)
            if n_ruido:
                W = np.vstack([W, trace[cand[:, None] + ar[None, :]]])
                L = np.concatenate([L, np.full(n_ruido, -1)])
                F = np.concatenate([F, cand])
                G = np.concatenate([G, cand])

        o = np.argsort(F, kind="stable")
        W, L, F, G = W[o], L[o], F[o], G[o]

        carpeta = os.path.join(a.out, f"canal_{ch}")
        os.makedirs(carpeta, exist_ok=True)
        pd.DataFrame(W).to_csv(os.path.join(carpeta, "spikes.csv"), header=False, index=False, float_format="%.3f")
        pd.DataFrame({"label": L}).to_csv(os.path.join(carpeta, "labels.csv"), header=False, index=False)
        pd.DataFrame({"spike_frame": F, "spike_time_s": F / fs, "gt_frame": G}).to_csv(
            os.path.join(carpeta, "times.csv"), index=False)
        pd.DataFrame(filas).to_csv(os.path.join(carpeta, "unidades.csv"), index=False)

        pct = 100.0 * (L != -1).mean()
        print(f"[{i}/{len(canales)}] canal {ch}: {len(filas)} unidades, {len(L)} spikes, {pct:.1f}% con etiqueta")
        resumen.append({"canal": ch, "estado": "ok", "n_unidades": len(filas), "n_spikes": len(L),
                        "n_ruido": n_ruido, "pct_con_label": round(pct, 2),
                        "unidades": " ".join(str(f["unidad"]) for f in filas)})
        del trace

    pd.DataFrame(resumen).to_csv(os.path.join(a.out, "resumen.csv"), index=False)
    ok = sum(1 for r in resumen if r["estado"] == "ok")
    print(f"Listo. {ok} de {len(canales)} canales guardados en {a.out} (ver resumen.csv)")


if __name__ == "__main__":
    main()
