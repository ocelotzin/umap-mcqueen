//! Sonda: ¿cuántas unidades separa DenStream en el encaje de un modelo real, y
//! con qué `macro_epsilon`?
//!
//! Proyecta las formas de onda con el modelo guardado (igual que `tdc_umap`:
//! normalización de entrada, `transform`, normalización del encaje), las pasa
//! por DenStream en lotes y, para varios `macro_epsilon` y `min_pts`, cuenta las
//! unidades y su tamaño. También lista la distancia de cada centroide de
//! micro-cluster a su vecino más cercano: el `macro_epsilon` bueno une los
//! centroides de una misma neurona (distancias cortas) y no salta los huecos
//! entre neuronas (distancias largas). Recuerda: `clump` compara distancias AL
//! CUADRADO; la sonda imprime ambas.
//!
//!   cargo run --release --example sonda_de_denstream -- <modelo base> <Waveforms.csv> [lote]

use std::collections::{HashMap, HashSet};
use std::error::Error;

use clump::{DenStream, NOISE};
use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use umap_mcqueen::{configuracion, denstream, modelo};

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn lee_csv(ruta: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut r = ReaderBuilder::new().has_headers(false).from_path(ruta)?;
    let mut filas = Vec::new();
    for reg in r.records() {
        let reg = reg?;
        filas.push(reg.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()?);
    }
    Ok(filas)
}

/// Unidades al final: grupo de DBSCAN o micro aislado → nº de puntos.
fn unidades(ds: &DenStream, puntos: &[Vec<f32>]) -> Vec<usize> {
    let macro_ = ds.macro_cluster().unwrap_or_default();
    let micro = ds.predict_batch(&puntos.to_vec()).unwrap_or_default();
    let mut tam: HashMap<i64, usize> = HashMap::new();
    for m in micro {
        if m == NOISE {
            continue;
        }
        let clave = match macro_.get(m) {
            Some(&k) if k != NOISE => k as i64,
            _ => -(m as i64) - 2,
        };
        *tam.entry(clave).or_default() += 1;
    }
    let mut v: Vec<usize> = tam.into_values().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("uso: sonda_de_denstream <modelo base> <Waveforms.csv> [lote]".into());
    }
    let (base, csv) = (&args[1], &args[2]);
    let lote: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(200);

    let datos = lee_csv(csv)?;
    let n_dim = datos.first().ok_or("CSV vacío")?.len();
    let (m, man) = modelo::carga_verificada::<MyAutodiffBackend>(
        base, configuracion(man_semilla(base)), n_dim, Default::default())?;
    let ne = man.normalizacion_de_entrada.ok_or("el manifiesto no trae normalización de entrada")?;
    let ns = man.normalizacion.ok_or("el manifiesto no trae normalización del encaje")?;

    let mut puntos: Vec<Vec<f32>> = Vec::with_capacity(datos.len());
    for trozo in datos.chunks(lote) {
        let enc = ns.aplica(m.transform(ne.aplica(trozo.to_vec())?));
        puntos.extend(enc.iter().map(|f| f.iter().map(|&x| x as f32).collect::<Vec<f32>>()));
    }
    let fuera = puntos.iter().flatten().filter(|&&x| !(0.0..=1.0).contains(&x)).count();
    println!("{} formas de onda proyectadas · {:.1}% de coordenadas fuera de [0,1]\n",
             puntos.len(), 100.0 * fuera as f64 / (2 * puntos.len()) as f64);

    // Distancias entre centroides de micro-clusters, con los parámetros actuales.
    let mut ds = denstream::nuevo();
    for t in puntos.chunks(lote) {
        ds.update_batch(&t.to_vec())?;
    }
    let c = ds.centroids();
    let mut vecino: Vec<f32> = c.iter().enumerate().map(|(i, a)| {
        c.iter().enumerate().filter(|(j, _)| *j != i)
            .map(|(_, b)| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt())
            .fold(f32::INFINITY, f32::min)
    }).collect();
    vecino.sort_by(|a, b| a.total_cmp(b));
    println!("{} micro-clusters. Distancia al centroide vecino más cercano (ordenada):", c.len());
    println!("  {}\n", vecino.iter().map(|d| format!("{d:.3}")).collect::<Vec<_>>().join(" "));

    println!("{:>12} {:>9} {:>8} {:>6} {:>9}  tamaños", "macro_eps²", "distancia", "min_pts", "micro", "unidades");
    let mut vistos = HashSet::new();
    for (me, mp) in [(0.2f32, 3usize), (0.1, 2), (0.04, 2), (0.02, 2), (0.01, 2), (0.04, 3)] {
        if !vistos.insert((me.to_bits(), mp)) { continue; }
        let mut ds = DenStream::new(denstream::EPSILON, mp)
            .with_beta(denstream::BETA).with_lambda(denstream::LAMBDA)
            .with_mu(denstream::MU).with_macro_epsilon(me);
        for t in puntos.chunks(lote) {
            ds.update_batch(&t.to_vec())?;
        }
        let u = unidades(&ds, &puntos);
        let marca = if me == denstream::MACRO_EPSILON && mp == denstream::MIN_PTS { "  ← actual" }
                    else if me == 0.2 && mp == 3 { "  ← antes" } else { "" };
        println!("{:>12} {:>9.3} {:>8} {:>6} {:>9}  {:?}{marca}",
                 me, me.sqrt(), mp, ds.n_clusters(), u.len(), &u[..u.len().min(12)]);
    }
    Ok(())
}

/// La semilla no forma parte de la arquitectura, pero se usa la del manifiesto
/// para que `configuracion` sea exactamente la del entrenamiento.
fn man_semilla(base: &str) -> u64 {
    std::fs::read_to_string(modelo::RutasDelModelo::de(base).manifiesto).ok()
        .and_then(|t| serde_json::from_str::<modelo::Manifiesto>(&t).ok())
        .and_then(|m| m.semilla)
        .unwrap_or(9999)
}
