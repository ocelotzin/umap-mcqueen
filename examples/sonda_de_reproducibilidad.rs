//! ¿Dónde nace el no determinismo?
//!
//! Dos corridas idénticas del programa dan resultados distintos. No es la semilla
//! —`fast_umap` llama a `B::seed(&device, 9999)`— ni es `autotune` —se probó sin esa
//! caracteristica y sigue—. Esta sonda localiza en qué etapa aparece, sin salir del
//! proceso, que es lo que permite descartar causas externas.
//!
//! Tres preguntas, de la más barata a la más cara:
//!
//!   1. ¿`transform()` es determinista? Mismo modelo, mismos datos, dos veces.
//!   2. ¿Lo es `fit()` DENTRO del mismo proceso? Dos entrenamientos seguidos.
//!   3. ¿Cuánto varía el encaje entre entrenamientos, comparado con su propia escala?
//!
//! Si (1) es determinista y (2) no, el no determinismo está en el ENTRENAMIENTO y no
//! en la inferencia — y eso importa: un modelo guardado seguiria dando lo mismo
//! siempre, y bastaría con entrenar una vez.
//!
//!   cargo run --release --example sonda_de_reproducibilidad -- <Waveforms.csv>

use csv::ReaderBuilder;
use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use std::error::Error;
use std::fs::File;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

const TRAIN: usize = 400;
const PRUEBA: usize = 200;

fn lee(ruta: &str, n: usize, salta: usize) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut l = ReaderBuilder::new().has_headers(false).from_reader(File::open(ruta)?);
    let mut v = Vec::new();
    for (i, r) in l.records().enumerate() {
        if i < salta { continue; }
        v.push(r?.iter().map(|c| c.trim().parse::<f64>()).collect::<Result<Vec<f64>, _>>()?);
        if v.len() == n { break; }
    }
    Ok(v)
}

fn config() -> UmapConfig {
    UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams { n_neighbors: 15, metric: Metric::Euclidean, ..Default::default() },
        optimization: OptimizationParams {
            n_epochs: 200, learning_rate: 1e-3, patience: Some(50), verbose: false, ..Default::default()
        },
        ..Default::default()
    }
}

/// Máxima diferencia absoluta entre dos encajes, y la escala del primero.
fn compara(a: &[Vec<f64>], b: &[Vec<f64>]) -> (f64, f64) {
    let mut peor = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        for (p, q) in x.iter().zip(y) { peor = peor.max((p - q).abs()); }
    }
    let plano: Vec<f64> = a.iter().flatten().cloned().collect();
    let media = plano.iter().sum::<f64>() / plano.len() as f64;
    let sd = (plano.iter().map(|v| (v - media).powi(2)).sum::<f64>() / plano.len() as f64).sqrt();
    (peor, sd)
}

fn main() -> Result<(), Box<dyn Error>> {
    let ruta = std::env::args().nth(1).ok_or("uso: sonda_de_reproducibilidad <Waveforms.csv>")?;
    let entrena = lee(&ruta, TRAIN, 0)?;
    let prueba = lee(&ruta, PRUEBA, TRAIN)?;
    println!("datos: {} de entrenamiento, {} de prueba\n", entrena.len(), prueba.len());

    // --- 1. ¿transform() es determinista? -----------------------------------
    let m = fast_umap::Umap::<MyAutodiffBackend>::new(config()).fit(entrena.clone(), None);
    let t1 = m.transform(prueba.clone());
    let t2 = m.transform(prueba.clone());
    let (dif_t, escala_t) = compara(&t1, &t2);
    println!("1. MISMO modelo, transform() dos veces");
    println!("   máxima diferencia: {dif_t:.3e}   (escala del encaje: {escala_t:.3})");
    println!("   → inferencia {}\n", if dif_t == 0.0 { "DETERMINISTA" } else { "NO determinista" });

    // --- 2. ¿fit() es determinista dentro del mismo proceso? ----------------
    let e1 = m.embedding().clone();
    let m2 = fast_umap::Umap::<MyAutodiffBackend>::new(config()).fit(entrena.clone(), None);
    let e2 = m2.embedding().clone();
    let (dif_f, escala_f) = compara(&e1, &e2);
    println!("2. DOS entrenamientos, mismo proceso, mismos datos");
    println!("   máxima diferencia: {dif_f:.3e}   (escala del encaje: {escala_f:.3})");
    println!("   relativa a la escala: {:.1} %", 100.0 * dif_f / escala_f);
    println!("   → entrenamiento {}\n", if dif_f == 0.0 { "DETERMINISTA" } else { "NO determinista" });

    // --- 3. ¿y el modelo guardado? ------------------------------------------
    // Si (1) es determinista, un modelo entrenado UNA vez y reutilizado da siempre
    // lo mismo, y el no determinismo de (2) deja de importar en produccion.
    let t3 = m.transform(prueba.clone());
    let (dif_g, _) = compara(&t1, &t3);
    println!("3. El mismo modelo, una tercera proyección: {dif_g:.3e}");

    println!("\nLECTURA:");
    if dif_t == 0.0 && dif_f > 0.0 {
        println!("  El no determinismo está en el ENTRENAMIENTO, no en la inferencia.");
        println!("  Consecuencia: entrenar una vez y GUARDAR el modelo elimina el problema");
        println!("  en producción — que es justo para lo que sirve el manifiesto.");
    } else if dif_t > 0.0 {
        println!("  También la inferencia varía. Guardar el modelo NO bastaría.");
    } else {
        println!("  Las dos etapas son deterministas: el no determinismo viene de fuera");
        println!("  del proceso y hay que buscarlo en otra parte.");
    }
    Ok(())
}
