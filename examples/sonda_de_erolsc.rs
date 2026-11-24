//! Sonda: calibración de EROLSC sobre un encaje ya calculado, contra la verdad.
//!
//! Lee el encaje (`dim1,dim2`, con cabecera) y las etiquetas verdaderas (una
//! por línea, sin cabecera), los pasa por EROLSC en lotes con una rejilla de
//! parámetros y reporta, por combinación: grupos, fracción de ruido, ARI
//! contra la verdad sin el ruido de EROLSC y ARI contando el ruido como un
//! grupo más. Sólo cuentan las espigas posteriores a la ventana inicial.
//!
//!   cargo run --release --example sonda_de_erolsc -- encaje.csv labels.csv [lote=200]

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
    println!("{} espigas, {} unidades verdaderas, lotes de {lote}", puntos.len(), {
        let mut u = verdad.clone();
        u.sort();
        u.dedup();
        u.len()
    });
    println!("{:>5} {:>5} {:>4} {:>6} {:>5} | {:>6} {:>7} {:>8} {:>9}", "Td", "Tn", "M", "fusion", "C", "grupos", "ruido%", "ARI_sin", "ARI_todo");

    let rejilla_td = [2.0, 2.25, 2.5];
    let rejilla_tn = [0.2, 0.4, 0.5, 0.6, 0.7];
    let rejilla_m = [50usize, 75, 100];
    let rejilla_f = [1.0];
    let rejilla_c = [10usize];
    for &td in &rejilla_td {
        for &tn in &rejilla_tn {
            for &m in &rejilla_m {
                for &f in &rejilla_f {
                    for &c in &rejilla_c {
                        let p = Parametros {
                            umbral_distancia: td,
                            umbral_pertenencia: tn,
                            min_puntos: m,
                            factor_fusion: f,
                            c_inicial: c,
                            c_anomalias: c,
                            ..Default::default()
                        };
                        let mut e = Erolsc::new(p)?;
                        let mut et = Vec::with_capacity(puntos.len());
                        for l in puntos.chunks(lote) {
                            et.extend(e.actualiza_lote(l)?);
                        }
                        let validos: Vec<usize> = (0..et.len()).filter(|&i| et[i] != SIN_CLASIFICAR).collect();
                        let ruido = validos.iter().filter(|&&i| et[i] == RUIDO).count();
                        let (mut a1, mut b1) = (Vec::new(), Vec::new());
                        for &i in &validos {
                            if et[i] != RUIDO {
                                a1.push(et[i]);
                                b1.push(verdad[i]);
                            }
                        }
                        let a2: Vec<i64> = validos.iter().map(|&i| et[i]).collect();
                        let b2: Vec<i64> = validos.iter().map(|&i| verdad[i]).collect();
                        println!(
                            "{td:>5} {tn:>5} {m:>4} {f:>6} {c:>5} | {:>6} {:>7.1} {:>8.3} {:>9.3}",
                            e.n_grupos(),
                            100.0 * ruido as f64 / validos.len() as f64,
                            ari(&a1, &b1),
                            ari(&a2, &b2)
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
