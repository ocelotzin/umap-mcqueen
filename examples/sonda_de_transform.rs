//! Sonda: ¿`transform()` devuelve puntos en el mismo espacio que `embedding()`?
//!
//! Toda la premisa de `umap-mcqueen` —y el motivo de elegir un UMAP paramétrico—
//! es que `transform()` coloque datos nuevos en el encaje ya aprendido. Esta sonda
//! comprueba lo mínimo exigible: que `transform()` sobre los MISMOS datos de
//! entrenamiento reproduzca `embedding()`.
//!
//!   cargo run --release --example sonda_de_transform -- <Waveforms.csv>

use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use fast_umap::utils::normalize_data;
use std::error::Error;
use std::fs::File;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

const TRAIN_SIZE: usize = 400;

fn rango(v: &[Vec<f64>]) -> (f64, f64) {
    (
        v.iter().flatten().cloned().fold(f64::INFINITY, f64::min),
        v.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let ruta = std::env::args().nth(1).ok_or("uso: sonda_de_transform <Waveforms.csv>")?;
    let mut lector = ReaderBuilder::new().has_headers(false).from_reader(File::open(&ruta)?);
    let mut datos: Vec<Vec<f64>> = Vec::new();
    for r in lector.records() {
        datos.push(r?.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<f64>, _>>()?);
        if datos.len() == TRAIN_SIZE { break; }
    }
    let n_dim = datos[0].len();

    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams { n_neighbors: 15, metric: Metric::Euclidean, ..Default::default() },
        optimization: OptimizationParams {
            n_epochs: 200, learning_rate: 1e-3, patience: Some(50), verbose: false, ..Default::default()
        },
        ..Default::default()
    };
    println!("graph.normalized = {}", config.graph.normalized);

    let ajustado = fast_umap::Umap::<MyAutodiffBackend>::new(config).fit(datos.clone(), None);

    let encaje = ajustado.embedding().clone();
    let (e0, e1) = rango(&encaje);
    println!("\nembedding() de los datos de entrenamiento : rango [{e0:.4}, {e1:.4}]");

    // A. transform() sobre LOS MISMOS datos, tal cual (lo que hace main.rs hoy).
    let crudo = ajustado.transform(datos.clone());
    let (c0, c1) = rango(&crudo);
    println!("transform() de esos mismos datos, crudos : rango [{c0:.4}, {c1:.4}]");

    // B. transform() pasando antes por la MISMA normalizacion que usa `finalize`.
    let mut plano: Vec<f64> = datos.iter().flatten().cloned().collect();
    normalize_data(&mut plano, datos.len(), n_dim);
    let normalizados: Vec<Vec<f64>> = plano.chunks(n_dim).map(|c| c.to_vec()).collect();
    let con_norm = ajustado.transform(normalizados);
    let (n0, n1) = rango(&con_norm);
    println!("transform() tras normalize_data()        : rango [{n0:.4}, {n1:.4}]");

    let dif = |a: &[Vec<f64>], b: &[Vec<f64>]| -> f64 {
        a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
            .fold(0.0f64, f64::max)
    };
    println!("\nmaxima diferencia contra embedding():");
    println!("  transform(crudo)                 {:.4e}", dif(&crudo, &encaje));
    println!("  transform(normalize_data(datos)) {:.4e}", dif(&con_norm, &encaje));

    println!(
        "\n`finalize` normaliza los datos antes de `model.forward`; `transform_to_tensor`\n\
         no lo hace. La red se entrena sobre datos normalizados y en inferencia recibe\n\
         datos crudos."
    );
    Ok(())
}
