//! `umap_mcqueen`: el núcleo de umap-mcqueen como biblioteca.
//!
//! La CLI (`src/main.rs`) y el puente a tridesclous (`tdc_umap/`, el módulo de
//! Python `tdc_umap`) usan estos mismos módulos, así que la configuración de
//! UMAP, el manifiesto del modelo y la normalización existen una sola vez.

use fast_umap::prelude::*;

/// Guardado y carga **verificada** del modelo: pesos más manifiesto.
pub mod denstream;
pub mod modelo;
/// Normalización con referencia fija, de la entrada y del encaje.
pub mod normaliza;
/// Simulador de espigas (mezclas de t de Student) para probar sin datos.
pub mod spikesim;

//La semilla entra aquí porque es parte de la configuración del entrenamiento
//(`UmapConfig::seed`): siembra la inicialización de la red y el muestreo de
//aristas de cada época. Misma configuración, mismos datos y misma semilla dan
//los mismos pesos; el manifiesto la guarda para que se sepa cuál fue.
pub fn configuracion(semilla: u64) -> UmapConfig {
    UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        seed: semilla,
        graph: GraphParams {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 200,
            learning_rate: 1e-3,
            patience: Some(50),
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    }
}
