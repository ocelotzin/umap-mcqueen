//! Sonda: separación local (mezcla::separa_unidad) sobre etiquetas ya hechas.
//!
//! Lee las formas de onda de un canal y unas etiquetas (p. ej. las de la CLI con
//! --etiquetas), parte cada unidad con GMM + BIC + filtro temporal mirando la
//! sesión entera, y escribe las etiquetas resultantes. Sirve para comparar la
//! implementación en Rust con el análisis offline (y para ajustar parámetros)
//! sin pasar por el agrupador en línea. El tiempo es el orden de la fila.
//!
//!   cargo run --release --example sonda_de_separacion -- spikes.csv etiquetas.csv salida.csv

use std::error::Error;

use umap_mcqueen::mezcla::{separa_unidad, ParamsSeparacion};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [formas, etiquetas, salida] = args.as_slice() else {
        return Err("uso: sonda_de_separacion <spikes.csv> <etiquetas.csv> <salida.csv>".into());
    };
    let mut lector = csv::ReaderBuilder::new().has_headers(false).from_path(formas)?;
    let ondas: Vec<Vec<f32>> = lector
        .records()
        .map(|r| r.map_err(|e| e.into()).and_then(|r| r.iter().map(|c| c.trim().parse::<f32>().map_err(Into::into)).collect::<Result<Vec<f32>, Box<dyn Error>>>()))
        .collect::<Result<_, _>>()?;
    let etq: Vec<i64> = std::fs::read_to_string(etiquetas)?.lines().map(|l| l.trim().parse()).collect::<Result<_, _>>()?;
    if etq.len() != ondas.len() {
        return Err(format!("{} formas y {} etiquetas", ondas.len(), etq.len()).into());
    }
    let p = ParamsSeparacion::default();
    let tramo = (0, ondas.len() as u64 - 1);
    let mut unidades: Vec<i64> = etq.iter().copied().filter(|&e| e >= 0).collect();
    unidades.sort_unstable();
    unidades.dedup();
    let mut nueva = vec![-1i64; etq.len()];
    let mut siguiente = 0i64;
    for u in unidades {
        let idx: Vec<usize> = (0..etq.len()).filter(|&i| etq[i] == u).collect();
        let f: Vec<&[f32]> = idx.iter().map(|&i| ondas[i].as_slice()).collect();
        let t: Vec<u64> = idx.iter().map(|&i| i as u64).collect();
        let sub = separa_unidad(&f, &t, tramo, &p).unwrap_or_else(|| vec![0; idx.len()]);
        let m = sub.iter().max().map_or(0, |&s| s + 1) as i64;
        println!("unidad {u}: {} espigas → {m} subgrupo(s)", idx.len());
        for (&i, &s) in idx.iter().zip(&sub) {
            nueva[i] = siguiente + s as i64;
        }
        siguiente += m;
    }
    std::fs::write(salida, nueva.iter().map(|e| format!("{e}\n")).collect::<String>())?;
    Ok(())
}
