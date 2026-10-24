//! Los parámetros de DenStream, una sola vez para la CLI y para `tdc_umap`.
//!
//! **Las distancias de `clump` van al cuadrado.** `DenStream::new` usa
//! `SquaredEuclidean`, así que `EPSILON` y `MACRO_EPSILON` se comparan contra la
//! distancia euclídea **elevada al cuadrado** entre puntos del encaje normalizado
//! a [0,1]. El radio de un micro-cluster, en cambio, se compara sin elevar.
//!
//! Esto es lo que rompía la clasificación: con el `macro_epsilon` por defecto de
//! `clump` (2 × epsilon = 0.2) DBSCAN une centroides a √0.2 ≈ 0.45 de distancia,
//! casi media caja [0,1]², y encadena todas las unidades en un solo grupo.
//! Con `MACRO_EPSILON = 0.04` el alcance es √0.04 = 0.2. Ver
//! `examples/sonda_de_denstream.rs` y `tests/denstream.rs`.

use clump::{DenStream, NOISE};

/// Distancia² máxima para absorber un punto en un micro-cluster (√0.1 ≈ 0.32),
/// y radio máximo, sin elevar, del micro-cluster resultante (0.1).
pub const EPSILON: f32 = 0.1;
/// Vecinos que necesita un centroide para ser núcleo en el DBSCAN que agrupa
/// micro-clusters en unidades. Con 3, una neurona que ocupa uno o dos
/// micro-clusters nunca es núcleo.
pub const MIN_PTS: usize = 2;
pub const BETA: f64 = 0.5;
pub const LAMBDA: f64 = 0.01;
pub const MU: f64 = 1.0;
/// Distancia² entre centroides para que DBSCAN los ponga en la misma unidad
/// (√0.04 = 0.2 en el encaje normalizado).
pub const MACRO_EPSILON: f32 = 0.04;

/// DenStream con los parámetros de arriba.
pub fn nuevo() -> DenStream {
    DenStream::new(EPSILON, MIN_PTS)
        .with_beta(BETA)
        .with_lambda(LAMBDA)
        .with_mu(MU)
        .with_macro_epsilon(MACRO_EPSILON)
}

/// Unidades que hay ahora: grupos de DBSCAN más micro-clusters aislados (que
/// DBSCAN deja como ruido pero `tdc_umap` trata como unidad propia).
pub fn n_unidades(ds: &DenStream) -> usize {
    let etiquetas = match ds.macro_cluster() {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let grupos = etiquetas
        .iter()
        .filter(|&&k| k != NOISE)
        .collect::<std::collections::HashSet<_>>()
        .len();
    grupos + etiquetas.iter().filter(|&&k| k == NOISE).count()
}
