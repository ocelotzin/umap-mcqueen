//! Sonda: ¿qué promete y qué cumple `FittedUmap::save` / `::load`?
//!
//! No es una prueba de la suite: es una medición fechada para responder cuatro
//! preguntas concretas antes de construir nada encima.
//!
//!   1. ¿Qué fichero escribe `save()` de verdad? (la ruta que se le pasa, u otra)
//!   2. ¿`transform()` da lo mismo antes de guardar y después de cargar?
//!   3. ¿Qué devuelve `embedding()` en un modelo cargado?
//!   4. ¿Cargar con una configuración que no corresponde falla, o calla?
//!
//! Se corre con datos reales:
//!   cargo run --release --example sonda_de_guardado -- <ruta a Waveforms.csv>

use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use std::error::Error;
use std::fs::File;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

const TRAIN_SIZE: usize = 400;
const PRUEBA_SIZE: usize = 200;

fn lee_csv(ruta: &str, n: usize, salta: usize) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    // has_headers(false): estos CSV no tienen cabecera. El `ReaderBuilder::new()`
    // por defecto SI la asume, y por tanto se come la primera espiga.
    let mut lector = ReaderBuilder::new().has_headers(false).from_reader(File::open(ruta)?);
    let mut filas = Vec::new();
    for (i, r) in lector.records().enumerate() {
        if i < salta { continue; }
        let reg = r?;
        filas.push(reg.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<f64>, _>>()?);
        if filas.len() == n { break; }
    }
    Ok(filas)
}

fn config_base() -> UmapConfig {
    UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams { n_neighbors: 15, metric: Metric::Euclidean, ..Default::default() },
        optimization: OptimizationParams {
            n_epochs: 50, learning_rate: 1e-3, patience: Some(20), verbose: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let ruta = std::env::args().nth(1).ok_or("uso: sonda_de_guardado <Waveforms.csv>")?;

    let entrena = lee_csv(&ruta, TRAIN_SIZE, 0)?;
    let prueba = lee_csv(&ruta, PRUEBA_SIZE, TRAIN_SIZE)?;
    let n_dim = entrena[0].len();
    println!("datos: {} de entrenamiento, {} de prueba, {} dimensiones", entrena.len(), prueba.len(), n_dim);

    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config_base());
    let ajustado = umap.fit(entrena.clone(), None);
    println!("encaje de entrenamiento: {} x {}", ajustado.embedding().len(), ajustado.embedding()[0].len());

    let antes = ajustado.transform(prueba.clone());

    // --- 1. que fichero escribe ---
    let dir = std::env::temp_dir().join("sonda_umap");
    std::fs::create_dir_all(&dir)?;
    let pedido = dir.join("modelo.umap");
    ajustado.save(&pedido)?;
    println!("\n1. se pidio guardar en : {}", pedido.display());
    println!("   existe esa ruta exacta: {}", pedido.exists());
    for e in std::fs::read_dir(&dir)? {
        let e = e?;
        println!("   hay en el directorio  : {} ({} bytes)", e.file_name().to_string_lossy(), e.metadata()?.len());
    }

    // --- 2. fidelidad del viaje de ida y vuelta ---
    let device = Default::default();
    let cargado = FittedUmap::<MyAutodiffBackend>::load(&pedido, config_base(), n_dim, device)?;
    let despues = cargado.transform(prueba.clone());

    let mut peor: f64 = 0.0;
    for (a, d) in antes.iter().zip(despues.iter()) {
        for (x, y) in a.iter().zip(d.iter()) {
            peor = peor.max((x - y).abs());
        }
    }
    println!("\n2. maxima diferencia absoluta de transform() antes vs despues: {:.3e}", peor);
    println!("   veredicto: {}", if peor < 1e-6 { "identico" } else { "NO reproduce" });

    // --- 3. que devuelve embedding() tras cargar ---
    println!("\n3. embedding() del modelo cargado: {} filas", cargado.embedding().len());
    println!("   (el de entrenamiento tenia {} filas)", ajustado.embedding().len());

    // --- 4. cargar con una configuracion que no corresponde ---
    let mut torcida = config_base();
    torcida.graph.n_neighbors = 99;          // no toca la arquitectura de la red
    torcida.graph.metric = Metric::Cosine;   // tampoco
    match FittedUmap::<MyAutodiffBackend>::load(&pedido, torcida, n_dim, Default::default()) {
        Ok(m) => {
            let t = m.transform(prueba.clone());
            let mut dif: f64 = 0.0;
            for (a, d) in antes.iter().zip(t.iter()) {
                for (x, y) in a.iter().zip(d.iter()) { dif = dif.max((x - y).abs()); }
            }
            println!("\n4. cargar con n_neighbors=99 y metric=Cosine: ACEPTADO sin queja");
            println!("   diferencia con el original: {:.3e}  <- el modelo es el mismo,", dif);
            println!("   pero ahora dice llamarse de otra manera. Nadie avisa.");
        }
        Err(e) => println!("\n4. cargar con configuracion torcida: RECHAZADO ({e})"),
    }

    // --- 4b. y con una arquitectura que no corresponde ---
    let mut otra = config_base();
    otra.hidden_sizes = vec![64];            // esto SI toca la arquitectura
    match FittedUmap::<MyAutodiffBackend>::load(&pedido, otra, n_dim, Default::default()) {
        Ok(m) => {
            let t = m.transform(prueba.clone());
            let mut dif: f64 = 0.0;
            for (a, d) in antes.iter().zip(t.iter()) {
                for (x, y) in a.iter().zip(d.iter()) { dif = dif.max((x - y).abs()); }
            }
            println!("4b. cargar con hidden_sizes=[64] en vez de [128]: ACEPTADO");
            println!("    diferencia con el original: {dif:.3e}");
            println!("    hidden_sizes que ahora declara config(): {:?}", m.config().hidden_sizes);
            println!("    -> los pesos del disco ganan, pero config() miente sobre ellos.");
        }
        Err(e) => println!("4b. cargar con hidden_sizes=[64] en vez de [128]: rechazado ({e})"),
    }

    Ok(())
}
