//! Pruebas del agrupador (HDBSCAN por ventana + Mahalanobis por espiga). No
//! usan la GPU: trabajan directamente sobre puntos del encaje normalizado.
//!
//!   cargo test --release --test agrupador
//!   UMAP_ENCAJE_REAL=encaje.csv cargo test --release --test agrupador -- --ignored

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use umap_mcqueen::agrupador::{Agrupador, ParamsAgrupador};

/// Agrupador con la auditoría en este mismo hilo: resultados deterministas (en
/// asíncrono, cuándo entra cada auditoría depende de la velocidad de la máquina).
fn sincrono() -> Agrupador {
    Agrupador::new(ParamsAgrupador { auditoria_asincrona: false, ..Default::default() }).unwrap()
}

fn g(sd: f32) -> Normal<f32> {
    Normal::new(0.0, sd).unwrap()
}

/// Índice de Rand ajustado (1 = misma partición, ~0 = al azar).
fn ari(a: &[i64], b: &[i32]) -> f64 {
    let mut t: HashMap<(i64, i32), f64> = HashMap::new();
    let mut ra: HashMap<i64, f64> = HashMap::new();
    let mut rb: HashMap<i32, f64> = HashMap::new();
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
    (s - e) / (0.5 * (sa + sb) - e)
}

/// Pasa el flujo en lotes de 200 y devuelve (etiqueta, verdad) de cada espiga.
fn corre(agr: &mut Agrupador, datos: &[(Vec<f32>, i32)]) -> Vec<(i64, i32)> {
    let mut out = Vec::with_capacity(datos.len());
    for lote in datos.chunks(200) {
        let pts: Vec<Vec<f32>> = lote.iter().map(|d| d.0.clone()).collect();
        for (d, e) in lote.iter().zip(agr.etiqueta(&pts).unwrap()) {
            out.push((e, d.1));
        }
    }
    out
}

/// Fracción de las espigas de la unidad verdadera `k` (en `rango`) que llevan
/// su etiqueta más frecuente, sin contar ruido; y esa etiqueta.
fn fidelidad(r: &[(i64, i32)], k: i32, rango: std::ops::Range<usize>) -> (f64, i64) {
    let mut c: HashMap<i64, usize> = HashMap::new();
    for &(e, v) in &r[rango] {
        if v == k && e >= 0 {
            *c.entry(e).or_default() += 1;
        }
    }
    let total: usize = c.values().sum();
    let (&id, &n) = c.iter().max_by_key(|(_, n)| **n).unwrap_or((&-1, &0));
    (if total == 0 { 0.0 } else { n as f64 / total as f64 }, id)
}

fn ari_segunda_mitad(r: &[(i64, i32)]) -> f64 {
    let (p, v): (Vec<i64>, Vec<i32>) = r[r.len() / 2..].iter().filter(|x| x.1 >= 0).copied().unzip();
    ari(&p, &v)
}

#[test]
fn cuatro_nubes() {
    let mut rng = StdRng::seed_from_u64(7);
    let centros = [(0.15f32, 0.2f32), (0.8, 0.25), (0.5, 0.85), (0.3, 0.6)];
    let datos: Vec<(Vec<f32>, i32)> = (0..20_000)
        .map(|_| {
            let k = rng.gen_range(0..4);
            (vec![centros[k].0 + g(0.03).sample(&mut rng), centros[k].1 + g(0.03).sample(&mut rng)], k as i32)
        })
        .collect();
    let mut agr = sincrono();
    let r = corre(&mut agr, &datos);
    let a = ari_segunda_mitad(&r);
    println!("cuatro nubes: {} unidades, {} ids, ARI {a:.3}", agr.n_unidades(), agr.n_ids_emitidos());
    assert_eq!(agr.n_unidades(), 4);
    assert!(a > 0.95);
}

/// Banda larga junto a una nube densa, nube dispersa, arco, unidad del 3 % y
/// 3 % de ruido uniforme: formas excéntricas y densidades distintas.
#[test]
fn formas_excentricas_y_densidades_distintas() {
    for semilla in [1u64, 2] {
        let mut rng = StdRng::seed_from_u64(semilla);
        let mut datos: Vec<(Vec<f32>, i32)> = (0..30_000)
            .map(|_| {
                let u: f32 = rng.r#gen();
                if u < 0.03 {
                    (vec![rng.r#gen(), rng.r#gen()], -1)
                } else if u < 0.33 {
                    let t: f32 = rng.r#gen();
                    let w = g(0.012).sample(&mut rng);
                    (vec![0.08 + 0.6 * t - 0.4 * w, 0.15 + 0.3 * t + 0.9 * w], 0)
                } else if u < 0.58 {
                    (vec![0.76 + g(0.01).sample(&mut rng), 0.47 + g(0.01).sample(&mut rng)], 1)
                } else if u < 0.78 {
                    (vec![0.3 + g(0.055).sample(&mut rng), 0.55 + g(0.055).sample(&mut rng)], 2)
                } else if u < 0.97 {
                    let a = rng.r#gen::<f32>() * std::f32::consts::PI;
                    let r = 0.15 + g(0.008).sample(&mut rng);
                    (vec![0.72 + r * a.cos(), 0.68 + r * a.sin()], 3)
                } else {
                    (vec![0.88 + g(0.008).sample(&mut rng), 0.12 + g(0.008).sample(&mut rng)], 4)
                }
            })
            .collect();
        datos.shuffle(&mut rng);
        let mut agr = sincrono();
        let r = corre(&mut agr, &datos);
        let a = ari_segunda_mitad(&r);
        println!("excéntrico, semilla {semilla}: {} unidades, {} ids, ARI {a:.3}", agr.n_unidades(), agr.n_ids_emitidos());
        assert!(a > 0.9, "semilla {semilla}: ARI {a:.3}");
    }
}

/// Dos nubes unidas por un puente con el 10 % de los puntos.
#[test]
fn un_puente_no_une_dos_unidades() {
    let mut rng = StdRng::seed_from_u64(5);
    let datos: Vec<(Vec<f32>, i32)> = (0..20_000)
        .map(|_| {
            let u: f32 = rng.r#gen();
            if u < 0.45 {
                (vec![0.3 + g(0.03).sample(&mut rng), 0.5 + g(0.03).sample(&mut rng)], 0)
            } else if u < 0.9 {
                (vec![0.7 + g(0.03).sample(&mut rng), 0.5 + g(0.03).sample(&mut rng)], 1)
            } else {
                let t: f32 = rng.r#gen();
                (vec![0.3 + 0.4 * t, 0.5 + g(0.01).sample(&mut rng)], -1)
            }
        })
        .collect();
    let mut agr = sincrono();
    let r = corre(&mut agr, &datos);
    let (f0, id0) = fidelidad(&r, 0, 10_000..20_000);
    let (f1, id1) = fidelidad(&r, 1, 10_000..20_000);
    println!("puente: {} unidades · fidelidad {f0:.2} / {f1:.2}", agr.n_unidades());
    assert_ne!(id0, id1);
    assert!(f0 > 0.95 && f1 > 0.95);
}

/// Una neurona que dispara el 1 % de las veces junto a una grande tiene su
/// propia etiqueta. (Con la ventana de 10 000 espigas, una unidad necesita al
/// menos `MIN_CLUSTER_SIZE` = 50 espigas en ella: el 0.5 %.)
#[test]
fn la_unidad_pequena_no_la_absorbe_la_grande() {
    let mut rng = StdRng::seed_from_u64(9);
    let datos: Vec<(Vec<f32>, i32)> = (0..40_000)
        .map(|_| {
            if rng.r#gen::<f32>() < 0.01 {
                (vec![0.62 + g(0.01).sample(&mut rng), 0.5 + g(0.01).sample(&mut rng)], 1)
            } else {
                (vec![0.45 + g(0.04).sample(&mut rng), 0.5 + g(0.04).sample(&mut rng)], 0)
            }
        })
        .collect();
    let mut agr = sincrono();
    let r = corre(&mut agr, &datos);
    let (f0, grande) = fidelidad(&r, 0, 20_000..40_000);
    let (f1, pequena) = fidelidad(&r, 1, 20_000..40_000);
    println!("grande → {grande} ({f0:.2}), pequeña → {pequena} ({f1:.2})");
    assert_ne!(grande, pequena, "la unidad pequeña quedó dentro de la grande");
    assert!(f1 > 0.8);
}

/// Dos unidades que derivan 0.3 a lo largo de 60 000 espigas (como A y B en
/// los datos reales): cada una conserva su id todo el tiempo.
#[test]
fn la_deriva_no_cambia_los_ids() {
    let mut rng = StdRng::seed_from_u64(3);
    let n = 60_000;
    let datos: Vec<(Vec<f32>, i32)> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            if rng.r#gen::<f32>() < 0.6 {
                (vec![0.2 + 0.3 * t + g(0.03).sample(&mut rng), 0.3 + g(0.03).sample(&mut rng)], 0)
            } else {
                (vec![0.5 + g(0.03).sample(&mut rng), 0.8 - 0.3 * t + g(0.03).sample(&mut rng)], 1)
            }
        })
        .collect();
    let mut agr = sincrono();
    let r = corre(&mut agr, &datos);
    let (f0, a) = fidelidad(&r, 0, 2_000..n);
    let (f1, b) = fidelidad(&r, 1, 2_000..n);
    println!("deriva: fidelidad {f0:.3} / {f1:.3} · {} ids · {} ajustes", agr.n_ids_emitidos(), agr.n_ajustes());
    assert_ne!(a, b);
    assert!(f0 > 0.98 && f1 > 0.98);
    assert!(agr.n_ids_emitidos() <= 3);
}

/// Una neurona que aparece a mitad de la sesión: sus espigas quedan fuera de
/// todas las unidades, eso adelanta el reajuste, y recibe id propio en pocos
/// cientos de espigas.
#[test]
fn una_neurona_nueva_recibe_id_pronto() {
    let mut rng = StdRng::seed_from_u64(4);
    let n = 30_000;
    let aparece = 15_000;
    let datos: Vec<(Vec<f32>, i32)> = (0..n)
        .map(|i| {
            let u: f32 = rng.r#gen();
            if i >= aparece && u < 0.2 {
                (vec![0.8 + g(0.02).sample(&mut rng), 0.8 + g(0.02).sample(&mut rng)], 2)
            } else if u < 0.6 {
                (vec![0.3 + g(0.03).sample(&mut rng), 0.3 + g(0.03).sample(&mut rng)], 0)
            } else {
                (vec![0.6 + g(0.03).sample(&mut rng), 0.3 + g(0.03).sample(&mut rng)], 1)
            }
        })
        .collect();
    let mut agr = sincrono();
    let r = corre(&mut agr, &datos);
    // Primera espiga de la nueva que lleva una etiqueta.
    let primera = (aparece..n).find(|&i| r[i].1 == 2 && r[i].0 >= 0).expect("nunca recibió etiqueta");
    let (f, id) = fidelidad(&r, 2, primera..n);
    let (_, id0) = fidelidad(&r, 0, aparece..n);
    let (_, id1) = fidelidad(&r, 1, aparece..n);
    println!("nueva: etiquetada desde la espiga {} (+{} tras aparecer), id {id}, fidelidad {f:.3}",
             primera, primera - aparece);
    assert!(primera - aparece <= 2_000, "tardó {} espigas", primera - aparece);
    assert!(id != id0 && id != id1 && f > 0.95);
}

/// Con tu encaje real (no va en el repositorio): 3 unidades estables, como
/// HDBSCAN sobre PCA de las formas de onda crudas.
#[test]
#[ignore]
fn encaje_real() {
    let ruta = std::env::var("UMAP_ENCAJE_REAL").expect("define UMAP_ENCAJE_REAL");
    let puntos: Vec<(Vec<f32>, i32)> = std::fs::read_to_string(&ruta)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| (l.split(',').map(|x| x.trim().parse::<f32>().unwrap()).collect(), 0))
        .collect();
    for asincrono in [false] {
        let mut agr = Agrupador::new(ParamsAgrupador { auditoria_asincrona: asincrono, ..Default::default() }).unwrap();
        let t0 = std::time::Instant::now();
        let r = corre(&mut agr, &puntos);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut c: HashMap<i64, usize> = HashMap::new();
        for &(e, _) in &r[r.len() / 2..] {
            *c.entry(e).or_default() += 1;
        }
        let n: usize = c.values().sum();
        let grandes = c.iter().filter(|&(&e, &k)| e >= 0 && k as f64 >= 0.02 * n as f64).count();
        let ruido = *c.get(&-1).unwrap_or(&0) as f64 / n as f64;
        println!("{} espigas en {ms:.0} ms ({}) · {grandes} unidades ≥2 % · {} ids · {} ajustes · ruido {:.1} %",
                 puntos.len(), if asincrono { "asíncrono" } else { "síncrono" },
                 agr.n_ids_emitidos(), agr.n_ajustes(), 100.0 * ruido);
        let mut por_id: Vec<(i64, usize)> = c.iter().filter(|(e, _)| **e >= 0).map(|(&e, &k)| (e, k)).collect();
        por_id.sort_by(|a, b| b.1.cmp(&a.1));
        println!("   espigas por id (2.ª mitad): {:?} · propuestas: {:?} · uniones: {:?}", &por_id[..por_id.len().min(8)], agr.propuestas(), agr.uniones());
        assert!(grandes >= 3);
    }
}

/// Continuidad: sobre una sonda fija, la etiqueta casi no cambia de un lote al
/// siguiente, ni siquiera con deriva. (Con reajustes periódicos de HDBSCAN,
/// cada reajuste reescribía de golpe una parte grande de la sonda.) Se mide
/// desde la primera auditoría (lote 25): antes, el radio del arranque de 200
/// espigas se corrige una vez y mueve la frontera del ruido (~8 % de la sonda,
/// un solo lote; ver `aplica_auditoria`).
#[test]
fn continuidad_de_un_lote_al_siguiente() {
    let mut rng = StdRng::seed_from_u64(8);
    let n = 40_000;
    let datos: Vec<(Vec<f32>, i32)> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let u: f32 = rng.r#gen();
            if u < 0.5 {
                (vec![0.25 + 0.15 * t + g(0.03).sample(&mut rng), 0.3 + g(0.03).sample(&mut rng)], 0)
            } else if u < 0.8 {
                (vec![0.6 + g(0.04).sample(&mut rng), 0.6 + g(0.02).sample(&mut rng)], 1)
            } else {
                (vec![0.75 + g(0.02).sample(&mut rng), 0.25 + g(0.02).sample(&mut rng)], 2)
            }
        })
        .collect();
    let sonda: Vec<Vec<f32>> = datos.iter().step_by(40).map(|d| d.0.clone()).collect();
    let mut agr = sincrono();
    let mut anterior: Option<Vec<i64>> = None;
    let mut peor: f64 = 0.0;
    for (k, lote) in datos.chunks(200).enumerate() {
        let pts: Vec<Vec<f32>> = lote.iter().map(|d| d.0.clone()).collect();
        agr.etiqueta(&pts).unwrap();
        let ahora: Vec<i64> = sonda.iter().map(|x| agr.clasifica_punto(x)).collect();
        if let Some(a) = &anterior {
            if k > 25 {
                let c = a.iter().zip(&ahora).filter(|(x, y)| x != y).count() as f64 / ahora.len() as f64;
                peor = peor.max(c);
            }
        }
        anterior = Some(ahora);
    }
    println!("continuidad: peor cambio de un lote al siguiente {:.2} % · {} ids", 100.0 * peor, agr.n_ids_emitidos());
    assert!(peor < 0.03, "peor cambio {:.2} %", 100.0 * peor);
}

/// Una unidad pequeña que el arranque no llega a formar (pocas espigas suyas
/// entre las 200 primeras) y que la propagación pega a su vecina grande: la
/// auditoría ve que esa etiqueta es dueña de dos clusters y, tras varias
/// auditorías seguidas, la separa. Es el caso de una sesión que empieza a mitad
/// de la grabación.
#[test]
fn una_unidad_absorbida_al_arrancar_se_separa() {
    for semilla in [21u64, 22, 23] {
        let mut rng = StdRng::seed_from_u64(semilla);
        let n = 40_000;
        // La pequeña no está en el arranque (empieza en la espiga 300), como
        // cuando la sesión arranca en un tramo en que no dispara.
        let datos: Vec<(Vec<f32>, i32)> = (0..n)
            .map(|i| {
                if i >= 300 && rng.r#gen::<f32>() < 0.07 {
                    (vec![0.6 + g(0.012).sample(&mut rng), 0.5 + g(0.012).sample(&mut rng)], 1)
                } else {
                    (vec![0.45 + g(0.04).sample(&mut rng), 0.5 + g(0.04).sample(&mut rng)], 0)
                }
            })
            .collect();
        let mut agr = sincrono();
        let r = corre(&mut agr, &datos);
        let (_, grande) = fidelidad(&r, 0, n / 2..n);
        let (f, pequena) = fidelidad(&r, 1, n / 2..n);
        println!("semilla {semilla}: grande → {grande}, pequeña → {pequena} ({f:.2}) · separaciones {:?} · uniones {:?}",
                 agr.separaciones(), agr.uniones());
        assert_ne!(grande, pequena, "semilla {semilla}: la pequeña sigue dentro de la grande");
        assert!(f > 0.8);
    }
}
