//! Sonda: calibración de EROLSC sobre un encaje ya calculado, contra la verdad.
//!
//! Lee el encaje (`dim1,dim2`, con cabecera) y las etiquetas verdaderas (una
//! por línea, sin cabecera), los pasa por EROLSC en lotes con una rejilla de
//! parámetros y escribe en la salida estándar un CSV con una fila por
//! combinación: grupos, fracción de anomalías y ARI contra la verdad (las
//! anomalías de EROLSC cuentan como una clase más).
//!
//! Las etiquetas se arman igual que con `umap-mcqueen --etiquetas`: todas las
//! espigas, y las de la ventana inicial con la etiqueta de la inicialización.
//! Así la fila de los parámetros por defecto, sobre el encaje de la CLI con la
//! misma semilla, da el mismo ARI que `erolsc_labels.csv`.
//!
//! `comparacion_erolsc_dev_umap_hdbscan.Rmd` lee estas rejillas (una por canal,
//! `erolsc_rejilla.csv`) para elegir los parámetros de cada canal con los demás
//! canales: validación cruzada dejando un canal fuera.
//!
//!   cargo run --release --example sonda_de_erolsc -- encaje.csv labels.csv [lote=200] > erolsc_rejilla.csv
//!   cargo run --release --example sonda_de_erolsc -- encaje.csv labels.csv 200 500,1000,2000 > erolsc_rejilla_ventana.csv

#[path = "../src/erolsc.rs"]
#[allow(dead_code)]
mod erolsc;

use erolsc::{Erolsc, Parametros, RUIDO, SIN_CLASIFICAR};
use std::collections::HashMap;
use std::error::Error;

fn ari(a: &[i64], b: &[i64]) -> f64 {
    let n = a.len() as f64;
    if a.len() < 2 {
        return 0.0;
    }
    let mut tabla: HashMap<(i64, i64), f64> = HashMap::new();
    let mut fa: HashMap<i64, f64> = HashMap::new();
    let mut fb: HashMap<i64, f64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *tabla.entry((x, y)).or_default() += 1.0;
        *fa.entry(x).or_default() += 1.0;
        *fb.entry(y).or_default() += 1.0;
    }
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let indice: f64 = tabla.values().map(|&v| c2(v)).sum();
    let sa: f64 = fa.values().map(|&v| c2(v)).sum();
    let sb: f64 = fb.values().map(|&v| c2(v)).sum();
    let esperado = sa * sb / c2(n);
    let maximo = (sa + sb) / 2.0;
    if maximo == esperado { 0.0 } else { (indice - esperado) / (maximo - esperado) }
}

/// Etiquetas de todas las espigas, como las escribe la CLI con `--etiquetas`.
fn etiqueta(puntos: &[Vec<f64>], p: Parametros, lote: usize) -> Result<(Vec<i64>, usize), Box<dyn Error>> {
    let mut e = Erolsc::new(p)?;
    let mut todas: Vec<i64> = Vec::with_capacity(puntos.len());
    for l in puntos.chunks(lote) {
        todas.extend(e.actualiza_lote(l)?);
        if e.inicializado() {
            let base = todas.len() - e.n_ventana();
            for (k, (_, et)) in e.ventana().enumerate() {
                if todas[base + k] == SIN_CLASIFICAR {
                    todas[base + k] = et;
                }
            }
        }
    }
    for x in todas.iter_mut() {
        if *x == SIN_CLASIFICAR {
            *x = RUIDO;
        }
    }
    Ok((todas, e.n_grupos()))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("uso: sonda_de_erolsc <encaje.csv> <labels.csv> [lote=200]".into());
    }
    let lote: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(200);
    let mut puntos = Vec::new();
    for r in csv::Reader::from_path(&args[0])?.records() {
        let r = r?;
        puntos.push(r.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()?);
    }
    let verdad: Vec<i64> = std::fs::read_to_string(&args[1])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<i64>())
        .collect::<Result<_, _>>()?;
    if verdad.len() != puntos.len() {
        return Err(format!("{} puntos y {} etiquetas", puntos.len(), verdad.len()).into());
    }

    //Con un cuarto argumento (lista de ventanas, p. ej. 500,1000,2000) la rejilla
    //barre también el tamaño de la ventana, alrededor de los valores buenos de
    //Td, Tn y M, y la salida lleva una columna `ventana`.
    let ventanas: Option<Vec<usize>> = args
        .get(3)
        .map(|s| s.split(',').map(|v| v.trim().parse()).collect::<Result<_, _>>())
        .transpose()?;
    let (td_def, tn_def, m_def): (Vec<f64>, Vec<f64>, Vec<usize>) = if ventanas.is_some() {
        (vec![2.25, 2.5], vec![0.4, 0.5, 0.6], vec![50, 100, 200])
    } else {
        (vec![2.0, 2.25, 2.5, 3.0, 3.5], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], vec![20, 50, 100, 200, 400])
    };
    //Las rejillas se pueden cambiar sin recompilar: SONDA_TD, SONDA_TN y SONDA_M,
    //listas separadas por comas.
    fn de_entorno<T: std::str::FromStr>(var: &str, defecto: Vec<T>) -> Result<Vec<T>, Box<dyn Error>> {
        match std::env::var(var) {
            Ok(s) => s
                .split(',')
                .map(|v| v.trim().parse::<T>().map_err(|_| format!("{var}: «{v}» no es válido").into()))
                .collect(),
            Err(_) => Ok(defecto),
        }
    }
    let rej_td = de_entorno("SONDA_TD", td_def)?;
    let rej_tn = de_entorno("SONDA_TN", tn_def)?;
    let rej_m = de_entorno("SONDA_M", m_def)?;
    let lista_v = ventanas.clone().unwrap_or_else(|| vec![Parametros::default().ventana]);

    if ventanas.is_some() {
        print!("ventana,");
    }
    println!("td,tn,m,fusion,c,grupos,ruido,ari,segundos");
    for &v in &lista_v {
        for &td in &rej_td {
            for &tn in &rej_tn {
                for &m in &rej_m {
                    if m >= v {
                        continue;
                    }
                    let p = Parametros {
                        ventana: v,
                        umbral_distancia: td,
                        umbral_pertenencia: tn,
                        min_puntos: m,
                        ..Default::default()
                    };
                    let (f, c) = (p.factor_fusion, p.c_inicial);
                    let t0 = std::time::Instant::now();
                    let (et, grupos) = etiqueta(&puntos, p, lote)?;
                    let seg = t0.elapsed().as_secs_f64();
                    let ruido = et.iter().filter(|&&x| x == RUIDO).count() as f64 / et.len() as f64;
                    if ventanas.is_some() {
                        print!("{v},");
                    }
                    println!("{td},{tn},{m},{f},{c},{grupos},{ruido:.5},{:.6},{seg:.2}", ari(&et, &verdad));
                }
            }
        }
    }
    Ok(())
}
