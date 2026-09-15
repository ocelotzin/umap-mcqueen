//! Sonda: ¿es comparable el espacio en el que trabaja DenStream de un lote al siguiente?
//!
//! `main.rs` normaliza CADA lote con el minimo y el maximo DE ESE LOTE, y luego
//! alimenta DenStream con el resultado. Si el minimo y el maximo se mueven entre
//! lotes, el mismo punto del encaje cae en coordenadas distintas segun el lote en
//! que llegue — y el `epsilon` de DenStream deja de significar una distancia fija.
//!
//! Esta sonda no opina: mide el minimo y el maximo lote a lote, y mide cuanto se
//! desplaza un punto fijo al normalizarse con cada uno de ellos.
//!
//!   cargo run --release --example sonda_de_normalizacion -- <Waveforms.csv>

use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use std::error::Error;
use std::fs::File;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

const TRAIN_SIZE: usize = 400;
const BATCH_SIZE: usize = 200;
const N_LOTES: usize = 12;

/// La funcion `normalize` de `main.rs`, verbatim, para medir lo que de verdad corre.
fn normalize(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let min = data.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    data.iter()
        .map(|fila| fila.iter().map(|&x| if range == 0.0 { 0.0 } else { (x - min) / range }).collect())
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let ruta = std::env::args().nth(1).ok_or("uso: sonda_de_normalizacion <Waveforms.csv>")?;
    let mut lector = ReaderBuilder::new().has_headers(false).from_reader(File::open(&ruta)?);
    let mut registros = lector.records().enumerate();

    let mut entrena: Vec<Vec<f64>> = Vec::new();
    for (_, r) in &mut registros {
        entrena.push(r?.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<f64>, _>>()?);
        if entrena.len() == TRAIN_SIZE { break; }
    }

    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams { n_neighbors: 15, metric: Metric::Euclidean, ..Default::default() },
        optimization: OptimizationParams {
            n_epochs: 200, learning_rate: 1e-3, patience: Some(50), verbose: false, ..Default::default()
        },
        ..Default::default()
    };
    let ajustado = fast_umap::Umap::<MyAutodiffBackend>::new(config).fit(entrena.clone(), None);

    // Un punto de referencia fijo: la primera forma de onda del entrenamiento.
    // Su sitio en el encaje NO cambia — lo que cambia es donde acaba tras normalizar.
    let ancla = ajustado.transform(vec![entrena[0].clone()])[0].clone();

    println!("lote   min del lote    max del lote    ancla normalizada (x, y)");
    let mut anclas: Vec<(f64, f64)> = Vec::new();
    let mut buffer: Vec<Vec<f64>> = Vec::new();
    let mut lote = 0usize;

    for (_, r) in registros {
        buffer.push(r?.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<f64>, _>>()?);
        if buffer.len() < BATCH_SIZE { continue; }

        let emb = ajustado.transform(buffer.clone());
        let min = emb.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max = emb.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rango = max - min;

        // El ancla, normalizada con ESTE lote (es lo que le pasaria si llegara aqui).
        let ax = (ancla[0] - min) / rango;
        let ay = (ancla[1] - min) / rango;
        anclas.push((ax, ay));
        println!("{lote:>4}   {min:>12.4}   {max:>12.4}    ({ax:.4}, {ay:.4})");

        // Comprobacion de que la funcion medida es la que corre.
        let _ = normalize(emb);

        buffer.clear();
        lote += 1;
        if lote == N_LOTES { break; }
    }

    let xs: Vec<f64> = anclas.iter().map(|a| a.0).collect();
    let ys: Vec<f64> = anclas.iter().map(|a| a.1).collect();
    let dx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let dy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - ys.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("\nEl ancla es UN SOLO punto, que no se mueve del encaje.");
    println!("Recorrido de su coordenada normalizada entre lotes:  x {dx:.4}   y {dy:.4}");
    println!("El epsilon de DenStream en main.rs es 0.1.");
    println!("Veredicto: {}", if dx.max(dy) > 0.1 {
        "el ancla se desplaza MAS que el propio epsilon -> el espacio no es comparable entre lotes"
    } else {
        "el desplazamiento queda por debajo del epsilon"
    });
    Ok(())
}
