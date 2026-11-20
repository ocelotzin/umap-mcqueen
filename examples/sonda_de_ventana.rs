//! Sonda: ¿la mezcla de la auditoría ve menos estructura por ajustarse sólo
//! sobre la ventana (espigas de voto claro)?
//!
//! Corre el agrupador como sonda_en_linea y, cada `auditoria` espigas, para
//! cada grupo ajusta `separa_unidad` sobre dos conjuntos del mismo tramo:
//!   ventana: las espigas del grupo en la ventana (lo que ve la auditoría);
//!   todas:   todas las espigas que el voto de primer nivel dio al grupo desde
//!            la más vieja de la ventana, claras o no.
//! Escribe filas `auditoria,grupo,conjunto,indice,subgrupo` (subgrupo -1 si no
//! partió) para medir contra la verdad de campo.
//!
//!   cargo run --release --example sonda_de_ventana -- encaje.csv spikes.csv salida.csv [auditoria=5000]

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;

use umap_mcqueen::agrupador::{Agrupador, ParamsAgrupador};
use umap_mcqueen::mezcla::{separa_unidad, ParamsSeparacion};

fn lee(ruta: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut l = csv::ReaderBuilder::new().has_headers(false).from_path(ruta)?;
    l.records()
        .map(|r| -> Result<Vec<f32>, Box<dyn Error>> { r?.iter().map(|c| Ok(c.trim().parse::<f32>()?)).collect() })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("uso: sonda_de_ventana <encaje.csv> <spikes.csv> <salida.csv> [auditoria=N]".into());
    }
    let encaje = lee(&args[0])?;
    let formas = lee(&args[1])?;
    let mut p = ParamsAgrupador { auditoria_asincrona: false, ..Default::default() };
    for a in &args[3..] {
        match a.split_once('=') {
            Some(("auditoria", v)) => p.auditoria = v.parse()?,
            _ => return Err(format!("argumento desconocido: {a}").into()),
        }
    }
    let cada = p.auditoria;
    let ps = ParamsSeparacion::default();
    let mut agr = Agrupador::new(p)?;
    // Grupo de primer nivel de cada espiga, con el estado justo antes de su lote.
    let mut grupo = vec![-1i64; encaje.len()];
    let mut salida = String::new();
    let mut n_aud = 0;
    for (k, (pts, fs)) in encaje.chunks(200).zip(formas.chunks(200)).enumerate() {
        for (j, x) in pts.iter().enumerate() {
            grupo[k * 200 + j] = agr.clasifica_punto(x);
        }
        agr.etiqueta_con_formas(pts, Some(fs))?;
        let hecho = (k + 1) * 200;
        if hecho % cada != 0 {
            continue;
        }
        n_aud += 1;
        let ventana = agr.ventana_grupos();
        let Some(t0) = ventana.iter().map(|v| v.0).min() else { continue };
        let tramo = (t0, hecho as u64 - 1);
        let mut por_ventana: HashMap<i64, Vec<usize>> = HashMap::new();
        for &(t, g) in &ventana {
            por_ventana.entry(g).or_default().push(t as usize);
        }
        let mut por_todas: HashMap<i64, Vec<usize>> = HashMap::new();
        for i in t0 as usize..hecho {
            if grupo[i] >= 0 {
                por_todas.entry(grupo[i]).or_default().push(i);
            }
        }
        let mut gs: Vec<i64> = por_ventana.keys().copied().collect();
        gs.sort_unstable();
        for g in gs {
            for (nombre, idx) in [("ventana", &por_ventana[&g]), ("todas", por_todas.get(&g).unwrap_or(&Vec::new()))] {
                let f: Vec<&[f32]> = idx.iter().map(|&i| formas[i].as_slice()).collect();
                let t: Vec<u64> = idx.iter().map(|&i| i as u64).collect();
                let sub = separa_unidad(&f, &t, tramo, &ps);
                for (n, &i) in idx.iter().enumerate() {
                    let s = sub.as_ref().map_or(-1, |v| v[n] as i64);
                    writeln!(salida, "{n_aud},{g},{nombre},{i},{s}")?;
                }
            }
        }
    }
    std::fs::write(&args[2], salida)?;
    Ok(())
}
