//! Sonda: subgrupos de la mezcla antes del filtro temporal, para estudiar el
//! filtro aparte (p. ej. cuándo el encadenamiento junta subgrupos que sí
//! coexisten).
//!
//!   offline: cada unidad de <etiquetas.csv> sobre la sesión entera
//!     cargo run --release --example sonda_de_cadena -- offline spikes.csv etiquetas.csv salida.csv
//!   linea: cada grupo de la ventana en cada auditoría del agrupador en línea
//!     cargo run --release --example sonda_de_cadena -- linea encaje.csv spikes.csv salida.csv
//!
//! Escribe filas `contexto,t0,t1,indice,subgrupo` (sólo contextos donde la
//! mezcla parte); el tiempo de una espiga es su fila.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;

use umap_mcqueen::agrupador::{Agrupador, ParamsAgrupador};
use umap_mcqueen::mezcla::{subgrupos_sin_filtro, ParamsSeparacion};

fn lee(ruta: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut l = csv::ReaderBuilder::new().has_headers(false).from_path(ruta)?;
    l.records()
        .map(|r| -> Result<Vec<f32>, Box<dyn Error>> { r?.iter().map(|c| Ok(c.trim().parse::<f32>()?)).collect() })
        .collect()
}

fn vuelca(sal: &mut String, ctx: usize, tramo: (u64, u64), idx: &[usize], ondas: &[Vec<f32>], p: &ParamsSeparacion) {
    let f: Vec<&[f32]> = idx.iter().map(|&i| ondas[i].as_slice()).collect();
    if let Some(sub) = subgrupos_sin_filtro(&f, p) {
        for (&i, s) in idx.iter().zip(sub) {
            let _ = writeln!(sal, "{ctx},{},{},{i},{s}", tramo.0, tramo.1);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [modo, a, b, salida] = args.as_slice() else {
        return Err("uso: sonda_de_cadena (offline spikes etiquetas | linea encaje spikes) salida".into());
    };
    let p = ParamsSeparacion::default();
    let mut sal = String::new();
    let mut ctx = 0;
    match modo.as_str() {
        "offline" => {
            let ondas = lee(a)?;
            let etq: Vec<i64> = std::fs::read_to_string(b)?.lines().map(|l| l.trim().parse()).collect::<Result<_, _>>()?;
            let mut us: Vec<i64> = etq.iter().copied().filter(|&e| e >= 0).collect();
            us.sort_unstable();
            us.dedup();
            for u in us {
                let idx: Vec<usize> = (0..etq.len()).filter(|&i| etq[i] == u).collect();
                vuelca(&mut sal, ctx, (0, ondas.len() as u64 - 1), &idx, &ondas, &p);
                ctx += 1;
            }
        }
        "linea" => {
            let (encaje, ondas) = (lee(a)?, lee(b)?);
            let pa = ParamsAgrupador { auditoria_asincrona: false, ..Default::default() };
            let cada = pa.auditoria;
            let mut agr = Agrupador::new(pa)?;
            for (k, (pts, fs)) in encaje.chunks(200).zip(ondas.chunks(200)).enumerate() {
                agr.etiqueta_con_formas(pts, Some(fs))?;
                let hecho = (k + 1) * 200;
                if hecho % cada != 0 {
                    continue;
                }
                let ventana = agr.ventana_grupos();
                let Some(t0) = ventana.iter().map(|v| v.0).min() else { continue };
                let mut por: HashMap<i64, Vec<usize>> = HashMap::new();
                for &(t, g) in &ventana {
                    por.entry(g).or_default().push(t as usize);
                }
                let mut gs: Vec<i64> = por.keys().copied().collect();
                gs.sort_unstable();
                for g in gs {
                    vuelca(&mut sal, ctx, (t0, hecho as u64 - 1), &por[&g], &ondas, &p);
                    ctx += 1;
                }
            }
        }
        _ => return Err(format!("modo desconocido: {modo}").into()),
    }
    std::fs::write(salida, sal)?;
    Ok(())
}
