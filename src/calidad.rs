//! Calidad de las unidades **sin verdad de campo**.
//!
//! Comparar un método contra otro (PCA offline, HDBSCAN sobre el encaje) dice si
//! coinciden, no si aciertan: ninguno de los dos es la verdad. Estas métricas no
//! dependen de ningún otro método de agrupamiento: miden cada unidad con la
//! fisiología (los tiempos de sus espigas) y con sus propias formas de onda.
//! Con ellas se puede preguntar «¿qué método produce más neuronas bien
//! aisladas?» sin que ninguno sea el juez.
//!
//! Por unidad:
//!
//! * **Violaciones del periodo refractario** y **contaminación** (Hill,
//!   Mehta y Kleinfeld, 2011). Una neurona no dispara dos veces en menos de
//!   ~2 ms; si una unidad tiene intervalos así, mezcla neuronas o ruido. Con V
//!   violaciones, N espigas, duración T y ventana Δ = t_r − t_c, la
//!   contaminación c cumple c(1 − c) = V·T / (2·Δ·N²).
//!   `t_c` es el **tiempo muerto del detector**: el adquisidor no puede registrar
//!   dos espigas más cerca que eso (en un Cerebus, ~la duración de la ventana de
//!   la forma de onda). Se estima del propio electrodo como el menor intervalo
//!   entre espigas consecutivas; si no deja ventana (t_c ≥ t_r), la contaminación
//!   no se puede medir y sale NaN.
//! * **Presencia**: fracción de intervalos de `bin_presencia_s` de la sesión en
//!   que la unidad tiene al menos una espiga.
//! * **SNR**: amplitud pico a pico de la forma media entre la desviación típica
//!   media de sus espigas alrededor de esa media.
//! * **Distancia de aislamiento** y **L-ratio** (Schmitzer-Torbert y col., 2005),
//!   en el espacio de componentes principales de las formas de onda crudas: qué
//!   tan lejos (Mahalanobis) quedan las espigas de las demás unidades.
//! * **Silueta** en ese mismo espacio, sobre una muestra.
//!
//! **Segmentos.** Si el tiempo retrocede de una fila a la siguiente, empieza un
//! segmento nuevo: un ensayo cuyos tiempos se cuentan desde su propio inicio (en
//! un registro real, 150 ensayos de ~9 s, cada uno desde cero). Los intervalos
//! entre espigas sólo se miden dentro de un segmento, la duración es la suma de
//! los segmentos y la presencia se mide en el tiempo encadenado. Ordenar todos
//! los tiempos juntos mezclaba ensayos: espigas de ensayos distintos a la misma
//! hora relativa contaban como violaciones (tiempo muerto «de 1 muestra», tasas
//! de ~2 000 Hz y ninguna unidad buena).
//!
//! **Qué decide si una unidad es buena: la fracción de intervalos cortos, no la
//! contaminación de Hill.** El estimador de Hill supone que las violaciones vienen
//! de espigas ajenas al azar, y divide por la ventana Δ entre el tiempo muerto y
//! el refractario. En un Cerebus el tiempo muerto es ~1.63 ms (49 muestras), así
//! que con refractario de 2 ms Δ = 0.37 ms: en un registro real, una unidad de
//! 1 302 espigas con sólo 4 intervalos cortos (0.3 %) daba contaminación 1.0. La
//! fracción de intervalos < refractario (`umbral_isi_cortos`, 1 % por defecto) no
//! depende de Δ y es el criterio habitual. La contaminación de Hill se sigue
//! reportando.
//!
//! Una unidad es **buena** si su fracción de intervalos cortos es menor que
//! `umbral_isi_cortos`, su presencia al menos `umbral_presencia` y está
//! aislada: distancia de aislamiento ≥ `umbral_aislamiento`, o, cuando no se
//! puede calcular (la unidad tiene más espigas que todas las demás juntas),
//! L-ratio < `umbral_l_ratio`.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ParamsCalidad {
    /// Frecuencia de muestreo de los tiempos, en Hz (Cerebus: 30 000).
    pub fs: f64,
    /// Periodo refractario, en ms.
    pub refractario_ms: f64,
    /// Fracción máxima de intervalos entre espigas por debajo del refractario.
    pub umbral_isi_cortos: f64,
    /// Ancho de los intervalos de la presencia, en s.
    pub bin_presencia_s: f64,
    pub umbral_contaminacion: f64,
    pub umbral_presencia: f64,
    pub umbral_aislamiento: f64,
    pub umbral_l_ratio: f64,
    /// Espigas como máximo para la silueta (cuesta O(n²)).
    pub max_silueta: usize,
}

impl Default for ParamsCalidad {
    fn default() -> Self {
        Self {
            fs: 30_000.0,
            refractario_ms: 2.0,
            umbral_isi_cortos: 0.01,
            bin_presencia_s: 60.0,
            umbral_contaminacion: 0.1,
            umbral_presencia: 0.9,
            umbral_aislamiento: 20.0,
            umbral_l_ratio: 0.1,
            max_silueta: 2_000,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalidadUnidad {
    pub etiqueta: i64,
    pub espigas: usize,
    pub tasa_hz: f64,
    pub violaciones: usize,
    /// violaciones / intervalos medidos (dentro de cada segmento).
    pub fraccion_isi_cortos: f64,
    /// Estimador de Hill; informativo (ver la nota de arriba).
    pub contaminacion: f64,
    pub presencia: f64,
    pub snr: f64,
    pub distancia_aislamiento: f64,
    pub l_ratio: f64,
    pub silueta: f64,
    pub buena: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalidadElectrodo {
    pub duracion_s: f64,
    /// Segmentos (ensayos) detectados: el tiempo vuelve atrás al empezar cada uno.
    pub segmentos: usize,
    /// Tiempo muerto del detector estimado (menor intervalo entre espigas
    /// consecutivas dentro de un segmento), ms.
    pub tiempo_muerto_ms: f64,
    /// Si la ventana [tiempo muerto, refractario] permite medir contaminación.
    pub contaminacion_medible: bool,
    pub unidades: Vec<CalidadUnidad>,
    pub n_buenas: usize,
    /// Fracción de TODAS las espigas que cae en unidades buenas.
    pub fraccion_espigas_buenas: f64,
}

// ------------------------------------------------------------ estadística

/// ln Γ(x) (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8,
        771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905,
        -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// P(a, x): función gamma incompleta inferior regularizada.
fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let (mut suma, mut termino, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..500 {
            ap += 1.0;
            termino *= x / ap;
            suma += termino;
            if termino.abs() < suma.abs() * 1e-14 {
                break;
            }
        }
        (suma.ln() - x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Fracción continua para Q(a, x) = 1 − P(a, x).
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-300;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Función de distribución acumulada de χ² con `k` grados de libertad.
pub fn chi2_cdf(x: f64, k: usize) -> f64 {
    gamma_p(k as f64 / 2.0, x / 2.0)
}

/// Inversa de una matriz simétrica definida positiva d × d (Gauss-Jordan).
fn invierte(m: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut a = m.to_vec();
    let mut inv = vec![0.0; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0;
    }
    for c in 0..d {
        let piv = (c..d).max_by(|&x, &y| a[x * d + c].abs().total_cmp(&a[y * d + c].abs()))?;
        if a[piv * d + c].abs() < 1e-300 {
            return None;
        }
        for k in 0..d {
            a.swap(c * d + k, piv * d + k);
            inv.swap(c * d + k, piv * d + k);
        }
        let p = a[c * d + c];
        for k in 0..d {
            a[c * d + k] /= p;
            inv[c * d + k] /= p;
        }
        for r in 0..d {
            if r != c {
                let f = a[r * d + c];
                if f != 0.0 {
                    for k in 0..d {
                        a[r * d + k] -= f * a[c * d + k];
                        inv[r * d + k] -= f * inv[c * d + k];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Contaminación de Hill y col. (2011): resuelve c(1 − c) = V·T / (2·Δ·N²).
/// Devuelve 1 si no hay solución real (la unidad está muy contaminada).
pub fn contaminacion_hill(violaciones: usize, n: usize, duracion_s: f64, ventana_s: f64) -> f64 {
    if n < 2 || duracion_s <= 0.0 || ventana_s <= 0.0 {
        return f64::NAN;
    }
    let r = violaciones as f64 * duracion_s / (2.0 * ventana_s * (n as f64) * (n as f64));
    if r >= 0.25 { 1.0 } else { (1.0 - (1.0 - 4.0 * r).sqrt()) / 2.0 }
}

// ------------------------------------------------------------- evaluación

/// Calidad de cada unidad de un etiquetado. `tiempos` en muestras (a `p.fs`),
/// `formas` las formas de onda crudas y `rasgos` sus componentes principales
/// (las mismas para todos los métodos que se comparen). La etiqueta −1 es
/// ruido y no se evalúa.
pub fn evalua(etiquetas: &[i64], tiempos: &[f64], formas: &[Vec<f64>], rasgos: &[Vec<f64>], p: &ParamsCalidad) -> CalidadElectrodo {
    let n = etiquetas.len();

    // Segmentos: empieza uno nuevo cada vez que el tiempo retrocede.
    let mut seg_de = vec![0usize; n];
    for i in 1..n {
        seg_de[i] = seg_de[i - 1] + usize::from(tiempos[i] < tiempos[i - 1]);
    }
    let n_seg = seg_de.last().map(|&s| s + 1).unwrap_or(0);
    let mut ini = vec![f64::INFINITY; n_seg];
    let mut fin = vec![f64::NEG_INFINITY; n_seg];
    for i in 0..n {
        let s = seg_de[i];
        ini[s] = ini[s].min(tiempos[i]);
        fin[s] = fin[s].max(tiempos[i]);
    }
    // Tiempo encadenado (en s): cada segmento empieza donde acabó el anterior.
    let mut desplazamiento = vec![0.0; n_seg];
    for s in 1..n_seg {
        desplazamiento[s] = desplazamiento[s - 1] + (fin[s - 1] - ini[s - 1]) / p.fs;
    }
    let continuo: Vec<f64> = (0..n).map(|i| desplazamiento[seg_de[i]] + (tiempos[i] - ini[seg_de[i]]) / p.fs).collect();
    let duracion_s = (0..n_seg).map(|s| (fin[s] - ini[s]) / p.fs).sum::<f64>().max(1e-9);

    // Tiempo muerto del detector: menor intervalo entre espigas consecutivas
    // de un mismo segmento (todas las unidades juntas).
    let mut muerto_s = f64::INFINITY;
    for i in 1..n {
        if seg_de[i] == seg_de[i - 1] {
            let dt = (tiempos[i] - tiempos[i - 1]) / p.fs;
            if dt > 0.0 {
                muerto_s = muerto_s.min(dt);
            }
        }
    }
    let muerto_s = if muerto_s.is_finite() { muerto_s } else { 0.0 };
    let refr_s = p.refractario_ms / 1000.0;
    let ventana_s = refr_s - muerto_s;
    let medible = ventana_s > 1e-4; // al menos 0.1 ms de ventana

    let mut por_unidad: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &e) in etiquetas.iter().enumerate() {
        if e >= 0 {
            por_unidad.entry(e).or_default().push(i);
        }
    }
    let mut ids: Vec<i64> = por_unidad.keys().copied().collect();
    ids.sort_unstable();

    // Silueta sobre una muestra regular de las espigas con unidad.
    let con_unidad: Vec<usize> = (0..n).filter(|&i| etiquetas[i] >= 0).collect();
    let paso = (con_unidad.len() / p.max_silueta.max(1)).max(1);
    let muestra: Vec<usize> = con_unidad.iter().step_by(paso).copied().collect();
    let silueta_de = silueta_por_unidad(&muestra, etiquetas, rasgos);

    let k = rasgos.first().map(|r| r.len()).unwrap_or(0);
    let mut unidades = Vec::new();
    for &e in &ids {
        let miembros = &por_unidad[&e];
        let m = miembros.len();

        // Refractario: intervalos entre espigas consecutivas de la unidad, sólo
        // dentro de un mismo segmento.
        let mut ts: Vec<(usize, f64)> = miembros.iter().map(|&i| (seg_de[i], continuo[i])).collect();
        ts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let intervalos = ts.windows(2).filter(|w| w[0].0 == w[1].0).count();
        let violaciones = ts.windows(2).filter(|w| w[0].0 == w[1].0 && w[1].1 - w[0].1 < refr_s).count();
        let fraccion_isi_cortos = if intervalos == 0 { f64::NAN } else { violaciones as f64 / intervalos as f64 };
        let contaminacion = if medible { contaminacion_hill(violaciones, m, duracion_s, ventana_s) } else { f64::NAN };

        // Presencia.
        let nbins = ((duracion_s / p.bin_presencia_s).ceil() as usize).max(1);
        let mut ocupado = vec![false; nbins];
        for &(_, t) in &ts {
            let b = ((t / p.bin_presencia_s) as usize).min(nbins - 1);
            ocupado[b] = true;
        }
        let presencia = ocupado.iter().filter(|&&o| o).count() as f64 / nbins as f64;

        // SNR.
        let d = formas[miembros[0]].len();
        let mut media = vec![0.0; d];
        for &i in miembros {
            for (a, &x) in media.iter_mut().zip(&formas[i]) {
                *a += x;
            }
        }
        media.iter_mut().for_each(|a| *a /= m as f64);
        let ptp = media.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - media.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut var = vec![0.0; d];
        for &i in miembros {
            for (j, &x) in formas[i].iter().enumerate() {
                var[j] += (x - media[j]).powi(2);
            }
        }
        let ruido = var.iter().map(|v| (v / (m.max(2) - 1) as f64).sqrt()).sum::<f64>() / d as f64;
        let snr = if ruido > 0.0 { ptp / ruido } else { f64::NAN };

        // Aislamiento en el espacio de rasgos (Mahalanobis con la covarianza de la unidad).
        let (distancia_aislamiento, l_ratio) = aislamiento(miembros, etiquetas, e, rasgos, k);

        let aislada = if distancia_aislamiento.is_finite() {
            distancia_aislamiento >= p.umbral_aislamiento
        } else {
            l_ratio.is_finite() && l_ratio < p.umbral_l_ratio
        };
        let buena = fraccion_isi_cortos.is_finite()
            && fraccion_isi_cortos < p.umbral_isi_cortos
            && presencia >= p.umbral_presencia
            && aislada;
        unidades.push(CalidadUnidad {
            etiqueta: e,
            espigas: m,
            tasa_hz: m as f64 / duracion_s,
            violaciones,
            fraccion_isi_cortos,
            contaminacion,
            presencia,
            snr,
            distancia_aislamiento,
            l_ratio,
            silueta: silueta_de.get(&e).copied().unwrap_or(f64::NAN),
            buena,
        });
    }
    let n_buenas = unidades.iter().filter(|u| u.buena).count();
    let en_buenas: usize = unidades.iter().filter(|u| u.buena).map(|u| u.espigas).sum();
    CalidadElectrodo {
        duracion_s,
        segmentos: n_seg,
        tiempo_muerto_ms: muerto_s * 1000.0,
        contaminacion_medible: medible,
        unidades,
        n_buenas,
        fraccion_espigas_buenas: if n == 0 { 0.0 } else { en_buenas as f64 / n as f64 },
    }
}

/// Distancia de aislamiento y L-ratio de la unidad `e`.
fn aislamiento(miembros: &[usize], etiquetas: &[i64], e: i64, rasgos: &[Vec<f64>], k: usize) -> (f64, f64) {
    let m = miembros.len();
    if k == 0 || m <= k {
        return (f64::NAN, f64::NAN);
    }
    let mut media = vec![0.0; k];
    for &i in miembros {
        for (a, &x) in media.iter_mut().zip(&rasgos[i]) {
            *a += x;
        }
    }
    media.iter_mut().for_each(|a| *a /= m as f64);
    let mut cov = vec![0.0; k * k];
    for &i in miembros {
        for a in 0..k {
            let da = rasgos[i][a] - media[a];
            for b in 0..k {
                cov[a * k + b] += da * (rasgos[i][b] - media[b]);
            }
        }
    }
    cov.iter_mut().for_each(|c| *c /= (m - 1) as f64);
    let traza: f64 = (0..k).map(|i| cov[i * k + i]).sum();
    for i in 0..k {
        cov[i * k + i] += 1e-9 * (traza / k as f64).max(1e-300);
    }
    let Some(inv) = invierte(&cov, k) else { return (f64::NAN, f64::NAN) };
    let d2 = |x: &[f64]| -> f64 {
        let dx: Vec<f64> = x.iter().zip(&media).map(|(a, b)| a - b).collect();
        (0..k).map(|a| dx[a] * (0..k).map(|b| inv[a * k + b] * dx[b]).sum::<f64>()).sum()
    };
    // Todas las espigas que no son de la unidad, incluido el ruido.
    let mut otras: Vec<f64> = etiquetas.iter().enumerate().filter(|&(_, &x)| x != e).map(|(i, _)| d2(&rasgos[i])).collect();
    if otras.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let l_ratio = otras.iter().map(|&x| 1.0 - chi2_cdf(x, k)).sum::<f64>() / m as f64;
    let id = if otras.len() >= m {
        otras.select_nth_unstable_by(m - 1, |a, b| a.total_cmp(b));
        otras[m - 1]
    } else {
        f64::NAN // más espigas en la unidad que fuera de ella: no está definida
    };
    (id, l_ratio)
}

/// Silueta media por unidad sobre la muestra (distancia euclídea en rasgos).
fn silueta_por_unidad(muestra: &[usize], etiquetas: &[i64], rasgos: &[Vec<f64>]) -> HashMap<i64, f64> {
    let mut res: HashMap<i64, (f64, usize)> = HashMap::new();
    let unidades: Vec<i64> = {
        let mut v: Vec<i64> = muestra.iter().map(|&i| etiquetas[i]).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if unidades.len() < 2 {
        return HashMap::new();
    }
    let dist = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
    for &i in muestra {
        let mut suma: HashMap<i64, (f64, usize)> = HashMap::new();
        for &j in muestra {
            if i != j {
                let s = suma.entry(etiquetas[j]).or_insert((0.0, 0));
                s.0 += dist(&rasgos[i], &rasgos[j]);
                s.1 += 1;
            }
        }
        let propia = etiquetas[i];
        let Some(&(sa, na)) = suma.get(&propia) else { continue };
        if na == 0 {
            continue;
        }
        let a = sa / na as f64;
        let b = suma
            .iter()
            .filter(|&(&u, &(_, nn))| u != propia && nn > 0)
            .map(|(_, &(s, nn))| s / nn as f64)
            .fold(f64::INFINITY, f64::min);
        if !b.is_finite() {
            continue;
        }
        let s = (b - a) / a.max(b);
        let r = res.entry(propia).or_insert((0.0, 0));
        r.0 += s;
        r.1 += 1;
    }
    res.into_iter().map(|(u, (s, c))| (u, s / c as f64)).collect()
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn chi2_conocidos() {
        assert!((chi2_cdf(9.2103, 2) - 0.99).abs() < 1e-4);
        assert!((chi2_cdf(9.4877, 4) - 0.95).abs() < 1e-4);
        assert!((chi2_cdf(1.0, 1) - 0.6827).abs() < 1e-3);
    }

    #[test]
    fn contaminacion_de_hill() {
        assert_eq!(contaminacion_hill(0, 1000, 600.0, 0.0015), 0.0);
        // Con r = V·T/(2·Δ·N²) = 0.09, c(1 − c) = 0.09 → c = 0.1 (V = 4 500).
        let n = 100_000usize;
        let v = (0.09 * 2.0 * 0.0015 * (n * n) as f64 / 600.0).round() as usize;
        assert!((contaminacion_hill(v, n, 600.0, 0.0015) - 0.1).abs() < 0.01);
        assert_eq!(contaminacion_hill(1_000_000, n, 600.0, 0.0015), 1.0);
    }

    /// Tren de espigas con periodo refractario de 2.5 ms (sin violaciones).
    fn tren(tasa: f64, dur: f64, semilla: u64, fs: f64) -> Vec<f64> {
        let mut s = semilla;
        let mut sig = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
        };
        let mut t = 0.0;
        let mut v = Vec::new();
        loop {
            t += 0.0025 + (-sig().ln()) / tasa;
            if t > dur {
                break;
            }
            v.push(t * fs);
        }
        v
    }

    /// Dos ensayos con tiempos desde cero: una neurona limpia en cada uno.
    /// Mezclarlos al ordenar fabricaba violaciones; por segmentos no hay.
    #[test]
    fn los_ensayos_no_se_mezclan() {
        let fs = 30_000.0;
        let a = tren(10.0, 30.0, 3, fs);
        let b = tren(10.0, 30.0, 4, fs);
        let tiempos: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
        let n = tiempos.len();
        let etiquetas = vec![0i64; n];
        let formas = vec![vec![0.0, -10.0, 4.0, 0.0]; n];
        let rasgos: Vec<Vec<f64>> = (0..n).map(|i| vec![(i % 7) as f64 * 0.01, (i % 5) as f64 * 0.01]).collect();
        let c = evalua(&etiquetas, &tiempos, &formas, &rasgos, &ParamsCalidad::default());
        assert_eq!(c.segmentos, 2);
        assert_eq!(c.unidades[0].violaciones, 0);
        assert!((c.duracion_s - 60.0).abs() < 1.0, "{}", c.duracion_s);
        assert!((c.unidades[0].tasa_hz - 10.0).abs() < 2.0);
    }

    #[test]
    fn una_unidad_limpia_y_una_mezcla() {
        let fs = 30_000.0;
        let a = tren(8.0, 600.0, 1, fs);
        let b = tren(8.0, 600.0, 2, fs);
        // Formas: A y B bien separadas en los rasgos.
        // En orden de tiempo, como llegan del adquisidor (un solo segmento).
        let mut eventos: Vec<(f64, usize, usize)> = a.iter().enumerate().map(|(j, &t)| (t, 0, j))
            .chain(b.iter().enumerate().map(|(j, &t)| (t, 1, j)))
            .collect();
        eventos.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut tiempos = Vec::new();
        let mut rasgos = Vec::new();
        let mut formas = Vec::new();
        let mut verdad = Vec::new();
        for (t, k, j) in eventos {
            let r = ((j * 7919 % 1000) as f64 / 1000.0 - 0.5) * 0.2;
            let c = if k == 0 { 0.0 } else { 5.0 };
            tiempos.push(t);
            rasgos.push(vec![c + r, r * 0.5, -r]);
            formas.push(vec![0.0, -10.0 * (1.0 + k as f64), 4.0, 0.0]);
            verdad.push(k as i64);
        }
        // Tiempo muerto 0: hay ventana para medir.
        let p = ParamsCalidad::default();
        let bien = evalua(&verdad, &tiempos, &formas, &rasgos, &p);
        assert_eq!(bien.n_buenas, 2, "{:#?}", bien.unidades);
        // Las dos neuronas en una sola unidad: aparecen violaciones.
        let juntas = vec![0i64; verdad.len()];
        let mal = evalua(&juntas, &tiempos, &formas, &rasgos, &p);
        assert!(mal.unidades[0].violaciones > 0);
        assert!(mal.unidades[0].contaminacion > bien.unidades[0].contaminacion);
    }
}
