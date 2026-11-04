//! Sonda: ¿cómo cambia la clasificación con los parámetros del agrupador
//! (propagación de etiquetas por vecindad) en tus datos?
//!
//! Proyecta las formas de onda con el modelo guardado, igual que `tdc_umap`, y
//! pasa el flujo completo, en lotes y en orden, por el `Agrupador` con varias
//! combinaciones de `min_cluster_size` (espigas mínimas de una unidad en la
//! ventana; se usa en el arranque y en los nacimientos) y `min_samples`. Para
//! cada una:
//!
//! * **unid ≥1 %**: etiquetas que se llevan al menos el 1 % de las espigas de la
//!   segunda mitad del flujo (la primera es calentamiento).
//! * **ids**: identificadores emitidos en toda la sesión. Muchos más que
//!   unidades = las unidades cambian de id con el tiempo.
//! * **% ruido**, **ajustes** de HDBSCAN, y **ms** que tardó todo el flujo.
//! * **ARI ref.**: cuánto se parece su etiquetado al de los valores por defecto
//!   (1 = idéntico). Si casi todas las filas están cerca de 1, la estructura es
//!   de los datos y no de los parámetros.
//!
//! Uso:
//!
//!   cargo run --release --example sonda_de_agrupador -- <modelo base> <Waveforms.csv> [--lote N] [--guardar encaje.csv]
//!   cargo run --release --example sonda_de_agrupador -- --puntos encaje.csv [--lote N]

use std::error::Error;
use std::time::Instant;

use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use umap_mcqueen::agrupador::{self, Agrupador, ParamsAgrupador};
use umap_mcqueen::{configuracion, modelo};

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// (min_cluster_size, min_samples) a probar. La primera es la de por defecto.
const COMBINACIONES: [(usize, usize); 9] = [
    (agrupador::MIN_CLUSTER_SIZE, agrupador::MIN_SAMPLES),
    (20, 5), (20, 10), (40, 5), (40, 25), (80, 10), (80, 25), (150, 25), (150, 50),
];

fn lee_csv(ruta: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut r = ReaderBuilder::new().has_headers(false).from_path(ruta)?;
    let mut filas = Vec::new();
    for reg in r.records() {
        let reg = reg?;
        filas.push(reg.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()?);
    }
    Ok(filas)
}

/// La semilla no es parte de la arquitectura, pero se usa la del manifiesto
/// para que `configuracion` sea exactamente la del entrenamiento.
fn semilla_de(base: &str) -> u64 {
    std::fs::read_to_string(modelo::RutasDelModelo::de(base).manifiesto)
        .ok()
        .and_then(|t| serde_json::from_str::<modelo::Manifiesto>(&t).ok())
        .and_then(|m| m.semilla)
        .unwrap_or(9999)
}

fn proyecta(base: &str, csv: &str, lote: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let datos = lee_csv(csv)?;
    let n_dim = datos.first().ok_or("CSV vacío")?.len();
    let (m, man) = modelo::carga_verificada::<MyAutodiffBackend>(
        base, configuracion(semilla_de(base)), n_dim, Default::default())?;
    let ne = man.normalizacion_de_entrada.ok_or("el manifiesto no trae normalización de entrada")?;
    let ns = man.normalizacion.ok_or("el manifiesto no trae normalización del encaje")?;
    let mut puntos = Vec::with_capacity(datos.len());
    for trozo in datos.chunks(lote) {
        let enc = ns.aplica(m.transform(ne.aplica(trozo.to_vec())?));
        puntos.extend(enc.iter().map(|f| f.iter().map(|&x| x as f32).collect::<Vec<f32>>()));
    }
    Ok(puntos)
}

/// Índice de Rand ajustado entre dos etiquetados (1 = la misma partición).
fn ari(a: &[i64], b: &[i64]) -> f64 {
    use std::collections::HashMap;
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

struct Fila {
    nombre: String,
    mayores: usize,
    ids: i64,
    ruido: f64,
    ajustes: u64,
    ms: f64,
    etiquetas: Vec<i64>,
}

fn corre(mcs: usize, ms: usize, puntos: &[Vec<f32>], lote: usize) -> Result<Fila, String> {
    let p = ParamsAgrupador { min_cluster_size: mcs, min_samples: ms, auditoria_asincrona: false, ..Default::default() };
    let mut agr = Agrupador::new(p)?;
    let mitad = puntos.len() / 2;
    let mut etiquetas = Vec::with_capacity(puntos.len() - mitad);
    let t0 = Instant::now();
    for (k, trozo) in puntos.chunks(lote).enumerate() {
        let e = agr.etiqueta(trozo)?;
        for (j, x) in e.into_iter().enumerate() {
            if k * lote + j >= mitad {
                etiquetas.push(x);
            }
        }
    }
    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    let mut conteo = std::collections::HashMap::<i64, usize>::new();
    for &e in &etiquetas {
        *conteo.entry(e).or_default() += 1;
    }
    let n = etiquetas.len().max(1) as f64;
    let mayores = conteo.iter().filter(|&(&e, &c)| e >= 0 && c as f64 / n >= 0.01).count();
    let ruido = *conteo.get(&agrupador::ETIQUETA_RUIDO).unwrap_or(&0) as f64 / n;
    Ok(Fila {
        nombre: format!("mcs {mcs:>3} · ms {ms:>2}"),
        mayores, ids: agr.n_ids_emitidos(), ruido, ajustes: agr.n_ajustes(), ms: ms_total, etiquetas,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opcion = |nombre: &str| args.iter().position(|a| a == nombre).and_then(|i| args.get(i + 1)).cloned();
    let lote: usize = opcion("--lote").map(|s| s.parse()).transpose()?.unwrap_or(200);

    let puntos: Vec<Vec<f32>> = if let Some(ruta) = opcion("--puntos") {
        lee_csv(&ruta)?.into_iter().map(|f| f.into_iter().map(|x| x as f32).collect()).collect()
    } else {
        let libres: Vec<&String> = args.iter().enumerate()
            .filter(|(i, a)| !a.starts_with("--") && !(*i > 0 && args[i - 1].starts_with("--")))
            .map(|(_, a)| a).collect();
        if libres.len() < 2 {
            return Err("uso: sonda_de_agrupador <modelo base> <Waveforms.csv> [--lote N] [--guardar encaje.csv]\n\
                        \x20     sonda_de_agrupador --puntos encaje.csv [--lote N]".into());
        }
        let p = proyecta(libres[0], libres[1], lote)?;
        if let Some(ruta) = opcion("--guardar") {
            let mut w = csv::WriterBuilder::new().has_headers(false).from_path(&ruta)?;
            for f in &p {
                w.write_record(f.iter().map(|x| x.to_string()))?;
            }
            w.flush()?;
            println!("Encaje guardado en {ruta} (repite con --puntos {ruta}).");
        }
        p
    };
    let d = ParamsAgrupador::default();
    println!("{} espigas · lotes de {lote} · ventana {} · {} vecinos · voto mínimo {}",
             puntos.len(), d.ventana, d.vecinos, d.voto_minimo);
    println!("Etiquetas medidas en la segunda mitad del flujo (la primera es calentamiento).\n");

    let mut filas = Vec::new();
    for &(mcs, ms) in &COMBINACIONES {
        filas.push(corre(mcs, ms, &puntos, lote)?);
    }
    println!("{:<18} {:>9} {:>5} {:>8} {:>8} {:>9} {:>8}", "", "unid ≥1%", "ids", "% ruido", "ajustes", "ms total", "ARI ref.");
    for (k, f) in filas.iter().enumerate() {
        let a = if k == 0 { "—".to_string() } else { format!("{:.3}", ari(&filas[0].etiquetas, &f.etiquetas)) };
        let marca = if k == 0 { "  ← por defecto" } else { "" };
        println!("{:<18} {:>9} {:>5} {:>7.1}% {:>8} {:>9.0} {:>8}{marca}",
                 f.nombre, f.mayores, f.ids, 100.0 * f.ruido, f.ajustes, f.ms, a);
    }
    let parecidas = filas[1..].iter().filter(|f| ari(&filas[0].etiquetas, &f.etiquetas) >= 0.9).count();
    println!("\n{parecidas} de {} combinaciones dan un etiquetado parecido al de por defecto (ARI ≥ 0.9).",
             filas.len() - 1);
    if parecidas * 2 >= filas.len() - 1 {
        println!("La estructura es estable: depende de los datos, no de los parámetros.");
    } else {
        println!("La estructura depende de los parámetros: mira la galería de formas de onda con las");
        println!("combinaciones que cambian el número de unidades antes de escoger.");
    }
    Ok(())
}
