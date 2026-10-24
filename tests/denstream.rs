//! DenStream con los parámetros de `umap_mcqueen::denstream` separa unidades
//! bien separadas en el encaje normalizado. No usa la GPU.
//!
//!   cargo test --release --test denstream
//!
//! Regresión del fallo en que todas las neuronas salían con la misma etiqueta:
//! con el `macro_epsilon` por defecto de `clump` (2 × epsilon, una distancia AL
//! CUADRADO) DBSCAN encadenaba los centroides de todas las unidades.

use clump::DenStream;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use umap_mcqueen::denstream;

/// Cuatro nubes en [0,1]², como un encaje normalizado con cuatro neuronas.
fn cuatro_nubes(n: usize, sd: f32, semilla: u64) -> Vec<Vec<f32>> {
    let centros = [(0.15f32, 0.2f32), (0.8, 0.25), (0.5, 0.85), (0.3, 0.6)];
    let mut rng = StdRng::seed_from_u64(semilla);
    let ruido = Normal::new(0.0f32, sd).unwrap();
    (0..n)
        .map(|_| {
            let c = centros[rng.gen_range(0..centros.len())];
            vec![c.0 + ruido.sample(&mut rng), c.1 + ruido.sample(&mut rng)]
        })
        .collect()
}

fn alimenta(ds: &mut DenStream, puntos: &[Vec<f32>]) {
    for lote in puntos.chunks(64) {
        ds.update_batch(&lote.to_vec()).unwrap();
    }
}

#[test]
fn separa_cuatro_unidades() {
    for sd in [0.02f32, 0.03, 0.05] {
        let mut ds = denstream::nuevo();
        alimenta(&mut ds, &cuatro_nubes(10_000, sd, 7));
        assert_eq!(denstream::n_unidades(&ds), 4, "sd={sd}: micro={}", ds.n_clusters());
    }
}

#[test]
fn el_macro_epsilon_por_defecto_de_clump_las_junta() {
    // Documenta el fallo: si esto deja de cumplirse, clump cambió su valor por
    // defecto o sus unidades, y hay que revisar src/denstream.rs.
    let mut ds = DenStream::new(0.1, 3).with_beta(0.5).with_lambda(0.01).with_mu(1.0);
    alimenta(&mut ds, &cuatro_nubes(10_000, 0.03, 7));
    assert!(denstream::n_unidades(&ds) < 4, "unidades={}", denstream::n_unidades(&ds));
}
