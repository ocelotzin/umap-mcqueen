//! Sonda: el agrupador en línea sobre un encaje ya calculado, sin GPU.
//!
//! Lee el encaje (el que escribe la CLI con --encaje) y las formas de onda del
//! mismo archivo, los pasa al agrupador en lotes como la CLI y escribe las
//! etiquetas. Imprime cada separación y unión aplicada. Sirve para probar
//! parámetros del agrupador en segundos, sin reentrenar UMAP.
//!
//!   cargo run --release --example sonda_en_linea -- encaje.csv spikes.csv salida.csv \
//!       [separar_tras=3] [auditoria=5000] [coexistencia=0.5] [sin_formas]

use std::error::Error;

use umap_mcqueen::agrupador::{Agrupador, ParamsAgrupador};

fn lee(ruta: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut l = csv::ReaderBuilder::new().has_headers(false).from_path(ruta)?;
    l.records()
        .map(|r| -> Result<Vec<f32>, Box<dyn Error>> { r?.iter().map(|c| Ok(c.trim().parse::<f32>()?)).collect() })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("uso: sonda_en_linea <encaje.csv> <spikes.csv> <salida.csv> [clave=valor ...] [sin_formas]".into());
    }
    let encaje = lee(&args[0])?;
    let formas = lee(&args[1])?;
    if encaje.len() != formas.len() {
        return Err(format!("{} puntos y {} formas", encaje.len(), formas.len()).into());
    }
    let mut p = ParamsAgrupador { auditoria_asincrona: false, ..Default::default() };
    let mut con_formas = true;
    let mut traza: Option<String> = None;
    for a in &args[3..] {
        match a.split_once('=') {
            Some(("separar_tras", v)) => p.separar_tras = v.parse()?,
            Some(("unir_tras", v)) => p.unir_tras = v.parse()?,
            Some(("auditoria", v)) => p.auditoria = v.parse()?,
            Some(("ventana", v)) => p.ventana = v.parse()?,
            Some(("vetar", v)) => p.vetar_uniones = v.parse()?,
            Some(("coexistencia", v)) => p.separacion.as_mut().ok_or("sin separación")?.coexistencia_minima = v.parse()?,
            Some(("traza", v)) => traza = Some(v.to_string()),
            None if a == "sin_formas" => con_formas = false,
            _ => return Err(format!("argumento desconocido: {a}").into()),
        }
    }
    let mut agr = Agrupador::new(p)?;
    let mut etiquetas = Vec::with_capacity(encaje.len());
    let (mut n_sep, mut n_uni) = (0, 0);
    // traza: grupo de primer nivel de cada espiga y si ese grupo tenía segundo nivel.
    let mut filas_traza = String::new();
    for (k, (pts, fs)) in encaje.chunks(200).zip(formas.chunks(200)).enumerate() {
        if traza.is_some() {
            let con_modelo: std::collections::HashSet<i64> = agr.segundo_nivel().into_iter().map(|(g, _)| g).collect();
            for x in pts {
                let g = agr.clasifica_punto(x);
                filas_traza.push_str(&format!("{g},{}\n", con_modelo.contains(&g) as u8));
            }
        }
        let e = if con_formas { agr.etiqueta_con_formas(pts, Some(fs))? } else { agr.etiqueta(pts)? };
        etiquetas.extend(e);
        let propuestas: Vec<_> = agr.propuestas().iter().filter(|p| matches!(p, umap_mcqueen::agrupador::Propuesta::Separar(_))).collect();
        if agr.separaciones().len() > n_sep {
            println!("espiga {:>6}: separación {:?}", (k + 1) * 200, &agr.separaciones()[n_sep..]);
            n_sep = agr.separaciones().len();
        }
        if agr.uniones().len() > n_uni {
            println!("espiga {:>6}: unión {:?}", (k + 1) * 200, &agr.uniones()[n_uni..]);
            n_uni = agr.uniones().len();
        }
        if (k + 1) * 200 % agr.params().auditoria == 0 {
            println!("espiga {:>6}: propuestas de separar {:?} · unidades {}", (k + 1) * 200, propuestas, agr.n_unidades());
        }
    }
    std::fs::write(&args[2], etiquetas.iter().map(|e| format!("{e}\n")).collect::<String>())?;
    if let Some(r) = traza {
        std::fs::write(r, filas_traza)?;
    }
    println!("ids emitidos {} · unidades {}", agr.n_ids_emitidos(), agr.n_unidades());
    Ok(())
}
