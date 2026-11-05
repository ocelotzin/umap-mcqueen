//! Sonda: ¿cuánto se parece la clasificación EN LÍNEA a una referencia?
//!
//! Corre el agrupador como en línea (`Agrupador`, lotes de 200, en orden) y lo
//! compara con una de dos referencias:
//!
//! **Formas de onda reales** (`--formas Waveforms.csv`): la referencia es un
//! spike sorting clásico OFFLINE, que ve toda la sesión de una vez y no usa
//! UMAP: PCA de las formas de onda crudas (3 componentes) y HDBSCAN. Evalúa la
//! cadena entera (encaje + agrupador). Además se compara con HDBSCAN offline
//! sobre el mismo encaje, que evalúa sólo el agrupador. Ninguna de las dos es
//! verdad de campo: son otro método con sus propios errores. Que coincidan
//! dice que el resultado es consistente, no que acierte.
//!
//! **Simulación** (`--simular K`): vectores de una mezcla de K componentes t de
//! Student (`spikesim`), cada uno con el componente del que salió. Ésa sí es
//! la verdad. Se proyectan con UMAP entrenado en los primeros vectores (como
//! la CLI) o, sin GPU, con PCA (`--proyeccion pca`, para probar el agrupador).
//!
//! Reporta: ARI global y por tramos de la sesión, tabla de confusión
//! (qué unidad en línea corresponde a cuál de referencia, y cuáles se partieron
//! o se fusionaron), ruido de cada lado y, en simulación, cuántos vectores
//! tarda cada unidad en recibir su etiqueta.
//!
//! Uso:
//!
//!   # con el encaje ya guardado (sin GPU):
//!   cargo run --release --example sonda_contra_referencia -- --formas Waveforms.csv --puntos encaje.csv
//!   # proyectando con el modelo:
//!   cargo run --release --example sonda_contra_referencia -- --formas Waveforms.csv --modelo modelos/s1
//!   # contra la verdad del simulador:
//!   cargo run --release --example sonda_contra_referencia -- --simular 5 --vectores 40000
//!   cargo run --release --example sonda_contra_referencia -- --simular 5 --proyeccion pca
//!
//! Opciones: --lote N (200) · --tramos N (5) · --semilla S (simulación) ·
//! --train N (vectores para entrenar UMAP en simulación, 2000) ·
//! --guardar-etiquetas archivo.csv (índice, en línea, referencia) ·
//! --entrenar N (con --formas: entrena UMAP con las primeras N formas de onda
//! del propio archivo, en vez de --puntos o --modelo) ·
//! --resumen-json archivo.json (las medidas, para `scripts/compara_carpeta.py`) ·
//! --tiempos STime.csv (tiempo de cada espiga, en muestras; activa la calidad
//! sin verdad de campo: violaciones refractarias, contaminación, presencia,
//! SNR, aislamiento; ver `umap_mcqueen::calidad`) · --fs Hz (30000) ·
//! --refractario-ms (2) · --voto-minimo, --percentil-radio, --factor-nacer
//! (parámetros del agrupador en línea; ver `umap_mcqueen::agrupador`).

use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use hdbscan::{Hdbscan, HdbscanHyperParams, NnAlgorithm};
use umap_mcqueen::agrupador::{Agrupador, ParamsAgrupador, ETIQUETA_RUIDO};
use umap_mcqueen::calidad::{self, CalidadElectrodo, ParamsCalidad};
use umap_mcqueen::normaliza::{Normalizador, NormalizadorDeEntrada};
use umap_mcqueen::spikesim::GeneradorMezclaT;
use umap_mcqueen::{configuracion, modelo};

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// Espigas sobre las que corre el HDBSCAN de referencia (el resto se asigna al
/// vecino más cercano): el árbol de este HDBSCAN cuesta O(n²).
const MUESTRA_REFERENCIA: usize = 20_000;

// ------------------------------------------------------------------ datos

fn lee_csv(ruta: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut r = ReaderBuilder::new().has_headers(false).from_path(ruta)?;
    let mut filas = Vec::new();
    for reg in r.records() {
        let reg = reg?;
        filas.push(reg.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()?);
    }
    Ok(filas)
}

fn semilla_de(base: &str) -> u64 {
    std::fs::read_to_string(modelo::RutasDelModelo::de(base).manifiesto)
        .ok()
        .and_then(|t| serde_json::from_str::<modelo::Manifiesto>(&t).ok())
        .and_then(|m| m.semilla)
        .unwrap_or(9999)
}

/// Proyecta con un modelo guardado, igual que `tdc_umap`.
fn proyecta_con_modelo(base: &str, datos: &[Vec<f64>], lote: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let n_dim = datos.first().ok_or("sin datos")?.len();
    let (m, man) = modelo::carga_verificada::<MyAutodiffBackend>(
        base, configuracion(semilla_de(base)), n_dim, Default::default())?;
    let ne = man.normalizacion_de_entrada.ok_or("el manifiesto no trae normalización de entrada")?;
    let ns = man.normalizacion.ok_or("el manifiesto no trae normalización del encaje")?;
    let mut out = Vec::with_capacity(datos.len());
    for t in datos.chunks(lote) {
        let e = ns.aplica(m.transform(ne.aplica(t.to_vec())?));
        out.extend(e.iter().map(|f| f.iter().map(|&x| x as f32).collect::<Vec<f32>>()));
    }
    Ok(out)
}

/// Entrena UMAP con los primeros `train` vectores y proyecta todos (como la CLI).
fn proyecta_entrenando(datos: &[Vec<f64>], train: usize, lote: usize, semilla: u64) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let entreno = datos[..train.min(datos.len())].to_vec();
    let m = fast_umap::Umap::<MyAutodiffBackend>::new(configuracion(semilla)).fit(entreno.clone(), None);
    let ne = NormalizadorDeEntrada::ajusta(&entreno).ok_or("normalización de entrada")?;
    let ns = Normalizador::ajusta(m.embedding()).ok_or("normalización del encaje")?;
    let mut out = Vec::with_capacity(datos.len());
    for t in datos.chunks(lote) {
        let e = ns.aplica(m.transform(ne.aplica(t.to_vec())?));
        out.extend(e.iter().map(|f| f.iter().map(|&x| x as f32).collect::<Vec<f32>>()));
    }
    Ok(out)
}

// ------------------------------------------------------------------- PCA

/// Valores y vectores propios de una matriz simétrica (Jacobi cíclico).
fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let fuera: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j]).sum();
        if fuera < 1e-18 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// Proyección en las `k` primeras componentes principales, ajustadas con
/// `ajuste` (filas) y aplicadas a `datos`, escaladas por la desviación de la 1.ª.
fn pca(ajuste: &[Vec<f64>], datos: &[Vec<f64>], k: usize) -> Vec<Vec<f32>> {
    let d = ajuste[0].len();
    let n = ajuste.len() as f64;
    let media: Vec<f64> = (0..d).map(|j| ajuste.iter().map(|f| f[j]).sum::<f64>() / n).collect();
    let mut cov = vec![0.0; d * d];
    for f in ajuste {
        for i in 0..d {
            let di = f[i] - media[i];
            for j in i..d {
                cov[i * d + j] += di * (f[j] - media[j]);
            }
        }
    }
    for i in 0..d {
        for j in i..d {
            cov[i * d + j] /= n - 1.0;
            cov[j * d + i] = cov[i * d + j];
        }
    }
    let (val, vec) = jacobi(cov, d);
    let mut orden: Vec<usize> = (0..d).collect();
    orden.sort_by(|&a, &b| val[b].total_cmp(&val[a]));
    let escala = val[orden[0]].max(1e-300).sqrt();
    datos
        .iter()
        .map(|f| {
            orden[..k]
                .iter()
                .map(|&c| ((0..d).map(|j| (f[j] - media[j]) * vec[j * d + c]).sum::<f64>() / escala) as f32)
                .collect()
        })
        .collect()
}

/// Lleva cada columna a [0, 1] con el mínimo y máximo de las primeras `n` filas
/// (la referencia fija, como el `Normalizador`).
fn a_unidad(p: &mut [Vec<f32>], n: usize) {
    let d = p[0].len();
    for j in 0..d {
        let (lo, hi) = p[..n].iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), f| (lo.min(f[j]), hi.max(f[j])));
        let r = (hi - lo).max(1e-12);
        for f in p.iter_mut() {
            f[j] = (f[j] - lo) / r;
        }
    }
}

// ------------------------------------------------------------- referencia

/// HDBSCAN offline: sobre una muestra regular y el resto al vecino más cercano.
fn referencia_hdbscan(puntos: &[Vec<f32>], mcs: usize, ms: usize) -> Result<Vec<i64>, Box<dyn Error>> {
    let paso = puntos.len().div_ceil(MUESTRA_REFERENCIA).max(1);
    let muestra: Vec<Vec<f32>> = puntos.iter().step_by(paso).cloned().collect();
    let mcs = (mcs / paso).max(5);
    let hp = HdbscanHyperParams::builder().min_cluster_size(mcs).min_samples(ms.min(mcs))
        .nn_algorithm(NnAlgorithm::KdTree).build();
    let cl = Hdbscan::new(&muestra, hp).cluster_par().map_err(|e| format!("HDBSCAN: {e:?}"))?;
    let d2 = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    Ok(puntos
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if i % paso == 0 {
                return cl[i / paso] as i64;
            }
            let (mut mejor, mut e) = (f32::INFINITY, -1i64);
            for (m, &c) in muestra.iter().zip(&cl) {
                let d = d2(x, m);
                if d < mejor {
                    mejor = d;
                    e = c as i64;
                }
            }
            e
        })
        .collect())
}

// ---------------------------------------------------------------- métricas

fn ari(a: &[i64], b: &[i64]) -> f64 {
    let mut t: HashMap<(i64, i64), f64> = HashMap::new();
    let mut ra: HashMap<i64, f64> = HashMap::new();
    let mut rb: HashMap<i64, f64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *t.entry((x, y)).or_default() += 1.0;
        *ra.entry(x).or_default() += 1.0;
        *rb.entry(y).or_default() += 1.0;
    }
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let s: f64 = t.values().map(|&x| c2(x)).sum();
    let sa: f64 = ra.values().map(|&x| c2(x)).sum();
    let sb: f64 = rb.values().map(|&x| c2(x)).sum();
    let e = sa * sb / c2(a.len() as f64);
    let d = 0.5 * (sa + sb) - e;
    if d == 0.0 { 1.0 } else { (s - e) / d }
}

/// ARI contando sólo lo que la referencia no considera ruido (el ruido en línea
/// cuenta como una clase más, así que penaliza).
fn ari_sin_ruido_ref(en_linea: &[i64], referencia: &[i64]) -> f64 {
    let (a, b): (Vec<i64>, Vec<i64>) =
        en_linea.iter().zip(referencia).filter(|&(_, &r)| r >= 0).map(|(&x, &r)| (x, r)).unzip();
    if a.is_empty() { f64::NAN } else { ari(&a, &b) }
}

fn nombre(e: i64) -> String {
    if e == ETIQUETA_RUIDO { "ruido".into() } else { e.to_string() }
}

/// Lo que `informe` mide, para `--resumen-json`.
#[derive(serde::Serialize)]
struct Medidas {
    referencia: String,
    ari_global: f64,
    ari_por_tramo: Vec<f64>,
    ruido_en_linea: f64,
    ruido_referencia: f64,
    unidades_referencia: usize,
    unidades_en_linea: usize,
    fusiones: Vec<String>,
    divisiones: Vec<String>,
    /// ARI después de juntar las unidades en línea que corresponden a la misma
    /// unidad de referencia: partir una unidad no cuenta como error.
    ari_tolerante: f64,
    /// Fracción de las espigas con unidad (en ambos lados) que caen en la unidad
    /// de referencia mayoritaria de su unidad en línea: mide las fusiones y las
    /// espigas mal asignadas, y no penaliza las divisiones.
    pureza: f64,
    /// Unidades en línea de al menos 0.5 % por unidad de referencia de ese tamaño.
    unidades_por_referencia: f64,
}

/// Pasa cada etiqueta en línea a la unidad de referencia con la que comparte más
/// espigas (sin contar el ruido de la referencia). Varias unidades en línea que
/// corresponden a la misma unidad de referencia quedan juntas: una división no
/// es un error. Una unidad en línea sin espigas con referencia conserva una
/// etiqueta propia; el ruido en línea sigue siendo ruido.
fn tolera_divisiones(en_linea: &[i64], referencia: &[i64]) -> (Vec<i64>, f64) {
    let mut conjunta: HashMap<(i64, i64), usize> = HashMap::new();
    for (&l, &r) in en_linea.iter().zip(referencia) {
        if l >= 0 && r >= 0 {
            *conjunta.entry((l, r)).or_default() += 1;
        }
    }
    let mut mayoritaria: HashMap<i64, (i64, usize)> = HashMap::new();
    for (&(l, r), &k) in &conjunta {
        let e = mayoritaria.entry(l).or_insert((r, 0));
        if k > e.1 || (k == e.1 && r < e.0) {
            *e = (r, k);
        }
    }
    // Etiquetas en línea sin correspondencia: lejos de cualquier id de referencia.
    let desplaza = 1_000_000_000;
    let tolerante: Vec<i64> = en_linea
        .iter()
        .map(|&l| if l < 0 { l } else { mayoritaria.get(&l).map(|&(r, _)| r).unwrap_or(desplaza + l) })
        .collect();
    let total: usize = conjunta.values().sum();
    let bien: usize = mayoritaria.values().map(|&(_, k)| k).sum();
    (tolerante, if total == 0 { f64::NAN } else { bien as f64 / total as f64 })
}

fn informe(titulo: &str, en_linea: &[i64], referencia: &[i64], tramos: usize, es_verdad: bool) -> Medidas {
    let n = en_linea.len();
    println!("\n=== {titulo} ===");
    let ruido_l = en_linea.iter().filter(|&&e| e < 0).count() as f64 / n as f64;
    let ruido_r = referencia.iter().filter(|&&e| e < 0).count() as f64 / n as f64;
    println!("ARI global (sin el ruido de la referencia): {:.3}", ari_sin_ruido_ref(en_linea, referencia));
    let (tolerante, pureza) = tolera_divisiones(en_linea, referencia);
    let ari_tolerante = ari_sin_ruido_ref(&tolerante, referencia);
    println!("ARI tolerando divisiones (unidades en línea de una misma unidad de referencia, juntas): {:.3} · pureza {:.3}",
             ari_tolerante, pureza);
    println!("Ruido: en línea {:.1} % · referencia {:.1} %", 100.0 * ruido_l, 100.0 * ruido_r);

    let t = n.div_ceil(tramos);
    let ari_tramos: Vec<f64> = (0..tramos)
        .map(|k| {
            let r = k * t..((k + 1) * t).min(n);
            ari_sin_ruido_ref(&en_linea[r.clone()], &referencia[r])
        })
        .collect();
    let por_tramo: Vec<String> = ari_tramos.iter().map(|a| format!("{a:.3}")).collect();
    println!("ARI por tramo de la sesión ({tramos} tramos de {t}): {}", por_tramo.join(" · "));

    // Tabla de confusión, desde cada unidad de referencia.
    let mut conjunta: HashMap<(i64, i64), usize> = HashMap::new();
    let mut tam_ref: HashMap<i64, usize> = HashMap::new();
    let mut tam_lin: HashMap<i64, usize> = HashMap::new();
    for (&l, &r) in en_linea.iter().zip(referencia) {
        *conjunta.entry((r, l)).or_default() += 1;
        *tam_ref.entry(r).or_default() += 1;
        *tam_lin.entry(l).or_default() += 1;
    }
    let mut refs: Vec<(i64, usize)> = tam_ref.iter().map(|(&e, &c)| (e, c)).collect();
    refs.sort_by(|a, b| (a.0 < 0).cmp(&(b.0 < 0)).then(b.1.cmp(&a.1)));
    let etiqueta_ref = if es_verdad { "verdad" } else { "referencia" };
    println!("\n{etiqueta_ref:>10} {:>8} {:>7}   → en línea (fracción de sus espigas)", "espigas", "%");
    for &(r, c) in &refs {
        if r >= 0 && (c as f64) < 0.005 * n as f64 {
            continue;
        }
        let mut destinos: Vec<(i64, usize)> =
            conjunta.iter().filter(|((rr, _), _)| *rr == r).map(|(&(_, l), &k)| (l, k)).collect();
        destinos.sort_by(|a, b| b.1.cmp(&a.1));
        let texto: Vec<String> = destinos
            .iter()
            .take(4)
            .map(|&(l, k)| format!("{}: {:.0} %", nombre(l), 100.0 * k as f64 / c as f64))
            .collect();
        println!("{:>10} {:>8} {:>6.1}%   → {}", nombre(r), c, 100.0 * c as f64 / n as f64, texto.join(" · "));
    }

    // Diagnóstico: unidades en línea que reúnen varias de referencia, y
    // unidades de referencia repartidas en varias en línea.
    // Fusión: dos (o más) unidades de referencia mandan la mayoría de sus
    // espigas a la misma unidad en línea. (Medido desde la referencia: una
    // unidad pequeña absorbida por una grande es una fracción mínima de la
    // grande, y medirlo desde la unidad en línea no la detectaba.)
    let mut destino_de: HashMap<i64, Vec<i64>> = HashMap::new();
    for &(r, c) in &refs {
        if r < 0 || (c as f64) < 0.005 * n as f64 {
            continue;
        }
        if let Some((&(_, l), &k)) = conjunta.iter().filter(|((rr, l), _)| *rr == r && *l >= 0).max_by_key(|&(_, &k)| k) {
            if k as f64 >= 0.5 * c as f64 {
                destino_de.entry(l).or_default().push(r);
            }
        }
    }
    let mut fusiones: Vec<String> = destino_de
        .iter()
        .filter(|(_, rs)| rs.len() >= 2)
        .map(|(l, rs)| {
            let mut rs = rs.clone();
            rs.sort_unstable();
            format!("{l} ← {rs:?}")
        })
        .collect();
    let mut divisiones = Vec::new();
    for &(r, c) in &refs {
        if r < 0 || (c as f64) < 0.005 * n as f64 {
            continue;
        }
        let partes: Vec<i64> = tam_lin
            .keys()
            .filter(|&&l| l >= 0 && conjunta.get(&(r, l)).copied().unwrap_or(0) as f64 >= 0.2 * c as f64)
            .copied()
            .collect();
        if partes.len() >= 2 {
            divisiones.push(format!("{r} → {:?}", partes));
        }
    }
    fusiones.sort();
    divisiones.sort();
    println!("\nFusiones (dos de {etiqueta_ref} con la mayoría de sus espigas en la misma unidad en línea): {}",
             if fusiones.is_empty() { "ninguna".into() } else { fusiones.join(" · ") });
    println!("Divisiones (una de {etiqueta_ref} repartida en dos en línea, ≥20 % cada una): {}",
             if divisiones.is_empty() { "ninguna".into() } else { divisiones.join(" · ") });

    let grandes = |m: &HashMap<i64, usize>| m.iter().filter(|&(&e, &c)| e >= 0 && c as f64 >= 0.005 * n as f64).count();
    let medidas = Medidas {
        referencia: titulo.to_string(),
        ari_global: ari_sin_ruido_ref(en_linea, referencia),
        ari_por_tramo: ari_tramos,
        ruido_en_linea: ruido_l,
        ruido_referencia: ruido_r,
        unidades_referencia: grandes(&tam_ref),
        unidades_en_linea: grandes(&tam_lin),
        fusiones: fusiones.clone(),
        divisiones: divisiones.clone(),
        ari_tolerante,
        pureza,
        unidades_por_referencia: if grandes(&tam_ref) == 0 { f64::NAN } else {
            grandes(&tam_lin) as f64 / grandes(&tam_ref) as f64
        },
    };
    if es_verdad {
        // Retraso de detección: desde la primera espiga de cada unidad verdadera
        // hasta la primera que recibe la etiqueta en línea que le corresponde.
        println!("\nRetraso de detección (vectores desde la primera espiga de la unidad):");
        for &(r, c) in &refs {
            if r < 0 {
                continue;
            }
            let mejor = conjunta
                .iter()
                .filter(|((rr, l), _)| *rr == r && *l >= 0)
                .max_by_key(|&(_, &k)| k)
                .map(|(&(_, l), _)| l);
            let Some(l) = mejor else {
                println!("  {r}: nunca recibió etiqueta ({c} espigas)");
                continue;
            };
            let primera = referencia.iter().position(|&x| x == r).unwrap();
            let detectada = (primera..n).find(|&i| referencia[i] == r && en_linea[i] == l);
            match detectada {
                Some(i) => println!("  {r} (en línea {l}): {} vectores (espiga {} de la sesión)", i - primera, i),
                None => println!("  {r}: nunca"),
            }
        }
    }
    medidas
}

// ----------------------------------------------------------------- calidad

fn f(x: f64, nd: usize) -> String {
    if x.is_finite() { format!("{x:.nd$}") } else { "—".into() }
}

/// Calidad de las unidades de cada método, sin que ninguno sea la referencia.
fn informe_calidad(calidades: &[(String, CalidadElectrodo)], p: &ParamsCalidad) {
    let Some((_, primero)) = calidades.first() else { return };
    println!("\n=== Calidad de las unidades (sin verdad de campo) ===");
    println!("Sesión de {:.0} s en {} segmento(s) · tiempo muerto del detector {:.2} ms · refractario {} ms{}",
             primero.duracion_s, primero.segmentos, primero.tiempo_muerto_ms, p.refractario_ms,
             if primero.contaminacion_medible { String::new() } else {
                 " · sin ventana entre tiempo muerto y refractario: la contaminación de Hill no se mide".into()
             });
    println!("Buena = intervalos < {} ms en menos del {} % · presencia ≥ {} · distancia de aislamiento ≥ {} (o L-ratio < {} si no se define)",
             p.refractario_ms, 100.0 * p.umbral_isi_cortos, p.umbral_presencia, p.umbral_aislamiento, p.umbral_l_ratio);
    println!("\n{:<18} {:>8} {:>7} {:>16}", "método", "unidades", "buenas", "espigas en buenas");
    for (n, c) in calidades {
        println!("{:<18} {:>8} {:>7} {:>15.1}%", n, c.unidades.len(), c.n_buenas, 100.0 * c.fraccion_espigas_buenas);
    }
    for (n, c) in calidades {
        println!("\n{n}:");
        println!("  {:>8} {:>8} {:>7} {:>6} {:>7} {:>8} {:>8} {:>6} {:>8} {:>8} {:>7}  {}",
                 "unidad", "espigas", "Hz", "viol.", "ISI<r", "contam.", "presenc.", "SNR", "aislam.", "L-ratio", "silueta", "");
        let mut us: Vec<_> = c.unidades.iter().collect();
        us.sort_by(|a, b| b.espigas.cmp(&a.espigas));
        for u in us {
            println!("  {:>8} {:>8} {:>7} {:>6} {:>6}% {:>8} {:>8} {:>6} {:>8} {:>8} {:>7}  {}",
                     u.etiqueta, u.espigas, f(u.tasa_hz, 1), u.violaciones, f(100.0 * u.fraccion_isi_cortos, 2),
                     f(u.contaminacion, 3),
                     f(u.presencia, 2), f(u.snr, 1), f(u.distancia_aislamiento, 1), f(u.l_ratio, 3),
                     f(u.silueta, 2), if u.buena { "✓ buena" } else { "" });
        }
    }
}

// --------------------------------------------------------------------- main

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opcion = |nombre: &str| args.iter().position(|a| a == nombre).and_then(|i| args.get(i + 1)).cloned();
    let num = |nombre: &str, def: usize| -> Result<usize, Box<dyn Error>> {
        Ok(opcion(nombre).map(|s| s.parse()).transpose()?.unwrap_or(def))
    };
    let lote = num("--lote", 200)?;
    let tramos = num("--tramos", 5)?.max(1);

    // (encaje, referencias)
    // Formas de onda crudas y sus 4 componentes principales, para la calidad.
    let mut para_calidad: Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = None;
    let (encaje, referencias): (Vec<Vec<f32>>, Vec<(String, Vec<i64>, bool)>) =
        if let Some(k) = opcion("--simular") {
            let k: usize = k.parse()?;
            let n = num("--vectores", 40_000)?;
            let train = num("--train", 2_000)?;
            let semilla = num("--semilla", 1)? as u64;
            let mut generador = GeneradorMezclaT::aleatorio(k, semilla);
            let (vectores, verdad): (Vec<Vec<f64>>, Vec<i64>) = (0..n)
                .map(|_| {
                    let (v, e) = generador.generar_con_etiqueta();
                    (v.to_vec(), e as i64)
                })
                .unzip();
            println!("Simulación: {n} vectores de {k} componentes (semilla {semilla}), entrenamiento con los primeros {train}.");
            let encaje = match opcion("--proyeccion").as_deref().unwrap_or("umap") {
                "pca" => {
                    let mut p = pca(&vectores[..train.min(n)], &vectores, 2);
                    a_unidad(&mut p, train.min(n));
                    p
                }
                "umap" => proyecta_entrenando(&vectores, train, lote, semilla)?,
                otra => return Err(format!("--proyeccion umap|pca, no {otra}").into()),
            };
            (encaje, vec![("Contra la VERDAD del simulador".to_string(), verdad, true)])
        } else {
            let formas = lee_csv(&opcion("--formas").ok_or("falta --formas Waveforms.csv (o --simular K)")?)?;
            let encaje: Vec<Vec<f32>> = if let Some(ruta) = opcion("--puntos") {
                lee_csv(&ruta)?.into_iter().map(|f| f.into_iter().map(|x| x as f32).collect()).collect()
            } else if let Some(base) = opcion("--modelo") {
                proyecta_con_modelo(&base, &formas, lote)?
            } else if let Some(n) = opcion("--entrenar") {
                // Entrena UMAP con las primeras N formas de onda de este mismo
                // archivo y proyecta todas: nada que guardar a mano.
                let n: usize = n.parse()?;
                let semilla = num("--semilla", 1)? as u64;
                println!("Entrenando UMAP con las primeras {n} formas de onda (semilla {semilla}).");
                proyecta_entrenando(&formas, n, lote, semilla)?
            } else {
                return Err("con --formas hace falta --puntos encaje.csv, --modelo base o --entrenar N".into());
            };
            if encaje.len() != formas.len() {
                return Err(format!("el encaje tiene {} filas y las formas {}: tienen que ser las mismas espigas, en el mismo orden",
                                   encaje.len(), formas.len()).into());
            }
            println!("{} espigas de {} muestras.", formas.len(), formas[0].len());
            let t0 = Instant::now();
            let paso = formas.len().div_ceil(MUESTRA_REFERENCIA).max(1);
            let ajuste: Vec<Vec<f64>> = formas.iter().step_by(paso).cloned().collect();
            let pcs = pca(&ajuste, &formas, 3);
            let ref_pca = referencia_hdbscan(&pcs, 500, 25)?;
            let ref_enc = referencia_hdbscan(&encaje, 500, 25)?;
            println!("Referencias offline calculadas en {:.1} s.", t0.elapsed().as_secs_f64());
            let rasgos: Vec<Vec<f64>> = pca(&ajuste, &formas, 4)
                .into_iter()
                .map(|f| f.into_iter().map(|x| x as f64).collect())
                .collect();
            para_calidad = Some((formas, rasgos));
            (encaje, vec![
                ("Contra spike sorting OFFLINE: PCA (3) de las formas de onda + HDBSCAN (toda la cadena)".to_string(), ref_pca, false),
                ("Contra HDBSCAN OFFLINE sobre el mismo encaje (sólo el agrupador)".to_string(), ref_enc, false),
            ])
        };

    // Parámetros del agrupador que se pueden variar desde fuera, para probar
    // políticas más o menos exigentes con las espigas dudosas.
    let mut pa = ParamsAgrupador { auditoria_asincrona: false, ..Default::default() };
    if let Some(v) = opcion("--voto-minimo") {
        pa.voto_minimo = v.parse()?;
    }
    if let Some(v) = opcion("--percentil-radio") {
        pa.percentil_radio = v.parse()?;
    }
    if let Some(v) = opcion("--factor-nacer") {
        pa.factor_nacer = v.parse()?;
    }
    println!("Agrupador: voto_minimo {} · percentil_radio {} · factor_nacer {}",
             pa.voto_minimo, pa.percentil_radio, pa.factor_nacer);
    let mut agr = Agrupador::new(pa)?;
    let t0 = Instant::now();
    let mut en_linea = Vec::with_capacity(encaje.len());
    for t in encaje.chunks(lote) {
        en_linea.extend(agr.etiqueta(t)?);
    }
    println!("En línea: {} espigas en lotes de {lote} en {:.1} s · {} ids · uniones {:?}",
             encaje.len(), t0.elapsed().as_secs_f64(), agr.n_ids_emitidos(), agr.uniones());

    let segundos_en_linea = t0.elapsed().as_secs_f64();
    let medidas: Vec<Medidas> = referencias
        .iter()
        .map(|(titulo, referencia, es_verdad)| informe(titulo, &en_linea, referencia, tramos, *es_verdad))
        .collect();
    // Calidad sin verdad de campo: necesita los tiempos de cada espiga.
    let mut calidades: Vec<(String, CalidadElectrodo)> = Vec::new();
    if let (Some(ruta_t), Some((formas, rasgos))) = (opcion("--tiempos"), &para_calidad) {
        let tiempos: Vec<f64> = lee_csv(&ruta_t)?.into_iter().map(|f| f.first().copied().unwrap_or(f64::NAN)).collect();
        if tiempos.len() != formas.len() {
            return Err(format!("{ruta_t} tiene {} filas y las formas {}: tienen que ser las mismas espigas",
                               tiempos.len(), formas.len()).into());
        }
        let pc = ParamsCalidad {
            fs: opcion("--fs").map(|s| s.parse()).transpose()?.unwrap_or(30_000.0),
            refractario_ms: opcion("--refractario-ms").map(|s| s.parse()).transpose()?.unwrap_or(2.0),
            ..Default::default()
        };
        let nombres = ["en_linea", "referencia_pca", "referencia_encaje"];
        let etiquetados: Vec<&[i64]> = std::iter::once(en_linea.as_slice())
            .chain(referencias.iter().map(|(_, r, _)| r.as_slice()))
            .collect();
        for (nombre, et) in nombres.iter().zip(etiquetados) {
            calidades.push((nombre.to_string(), calidad::evalua(et, &tiempos, formas, rasgos, &pc)));
        }
        informe_calidad(&calidades, &pc);
    }

    if let Some(ruta) = opcion("--resumen-json") {
        let cal: serde_json::Map<String, serde_json::Value> = calidades
            .iter()
            .map(|(n, c)| (n.clone(), serde_json::to_value(c).unwrap_or(serde_json::Value::Null)))
            .collect();
        let json = serde_json::json!({
            "espigas": en_linea.len(),
            "ids_emitidos": agr.n_ids_emitidos(),
            "uniones": agr.uniones().len(),
            "segundos_en_linea": segundos_en_linea,
            "comparaciones": medidas,
            "calidad": cal,
        });
        std::fs::write(&ruta, serde_json::to_string_pretty(&json)?)?;
    }

    if let Some(ruta) = opcion("--guardar-etiquetas") {
        let mut w = csv::WriterBuilder::new().from_path(&ruta)?;
        let mut cab = vec!["indice".to_string(), "en_linea".to_string()];
        cab.extend((0..referencias.len()).map(|i| format!("referencia_{}", i + 1)));
        w.write_record(&cab)?;
        for i in 0..en_linea.len() {
            let mut fila = vec![i.to_string(), en_linea[i].to_string()];
            fila.extend(referencias.iter().map(|(_, r, _)| r[i].to_string()));
            w.write_record(&fila)?;
        }
        w.flush()?;
        println!("\nEtiquetas guardadas en {ruta}.");
    }
    Ok(())
}
