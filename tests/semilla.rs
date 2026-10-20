//! La semilla manda: con la misma, dos entrenamientos dan el mismo modelo; con
//! otra, dan otro. Las dos mitades hacen falta: sin la segunda, una semilla que
//! no llegara a los generadores pasaría la primera.
//!
//! Dos niveles:
//! * `fast_umap` directo, con la misma configuración que `configuracion()` de
//!   `main.rs`: el encaje de entrenamiento, coordenada a coordenada;
//! * el binario entero, con datos simulados: los modelos guardados, cargados y
//!   aplicados a los mismos datos, y la semilla escrita en el manifiesto.
//!
//! Lo que NO se compara es el `.bin` byte a byte: burn sortea un identificador
//! por parámetro al crear la red (`ParamId::new`) y lo escribe en el fichero, así
//! que dos modelos con los mismos pesos difieren en esos ~50 bytes. Medido el
//! 2026-10-20: 51 bytes distintos de 13 022 con la misma semilla, 12 190 con otra.

use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use fast_umap::{FittedUmap, Umap};
use std::path::{Path, PathBuf};
use std::process::Command;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// Tres nubes separadas, construidas sin ningún generador aleatorio: así los
/// datos no pueden ser una fuente de variación entre corridas.
fn nubes() -> Vec<Vec<f64>> {
    let (por_nube, dim) = (60usize, 22usize);
    (0..3 * por_nube)
        .map(|i| {
            let nube = (i / por_nube) as f64;
            let t = (i % por_nube) as f64;
            (0..dim)
                .map(|j| 10.0 * nube * ((j % 3) as f64 + 1.0) + ((t * 7.0 + j as f64 * 3.0) % 5.0) * 0.1)
                .collect()
        })
        .collect()
}

/// La configuración de `main.rs::configuracion()`, con `verbose` apagado.
fn configuracion(semilla: u64) -> UmapConfig {
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
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn encaje(semilla: u64) -> Vec<Vec<f64>> {
    Umap::<MyAutodiffBackend>::new(configuracion(semilla)).fit(nubes(), None).into_embedding()
}

fn diferencia_maxima(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f64::max)
}

fn escala(a: &[Vec<f64>]) -> f64 {
    let todo = a.iter().flatten();
    let max = todo.clone().fold(f64::MIN, |m, &v| m.max(v));
    let min = todo.fold(f64::MAX, |m, &v| m.min(v));
    max - min
}

/// Una sola prueba, secuencial a propósito: `Backend::seed` siembra el
/// generador del dispositivo, que comparten todos los entrenamientos del
/// proceso. Dos entrenamientos a la vez en hilos distintos se resiembran el
/// generador el uno al otro a mitad del entrenamiento, y parece que la semilla
/// no hace nada.
#[test]
fn misma_semilla_mismo_encaje_y_semillas_distintas_encajes_distintos() {
    let a = encaje(7);
    let b = encaje(7);
    let c = encaje(8);
    let s = escala(&a);

    let d_misma = diferencia_maxima(&a, &b);
    assert_eq!(d_misma, 0.0, "dos entrenamientos con la semilla 7 difieren hasta {d_misma} (escala del encaje {s})");

    let d_otra = diferencia_maxima(&a, &c);
    assert!(
        d_otra > 0.01 * s,
        "las semillas 7 y 8 dan encajes a menos de {d_otra} (escala {s}): la semilla no está llegando a los generadores"
    );
}

// ── el binario entero ────────────────────────────────────────────────────────

/// Entrena el binario con datos simulados y devuelve el manifiesto que escribió.
fn entrena(dir: &Path, nombre: &str, semilla: u64) -> serde_json::Value {
    let base = dir.join(nombre);
    let salida = Command::new(env!("CARGO_BIN_EXE_umap-mcqueen"))
        .args([
            "--simular", "3", "--vectores-sim", "500", "--semilla-sim", "1",
            "--train", "300", "--lote", "100",
            "--semilla", &semilla.to_string(),
            "--modelo", base.to_str().unwrap(),
        ])
        .output()
        .expect("no se pudo lanzar umap-mcqueen");
    assert!(
        salida.status.success(),
        "umap-mcqueen falló:\n{}\n{}",
        String::from_utf8_lossy(&salida.stdout),
        String::from_utf8_lossy(&salida.stderr)
    );
    serde_json::from_str(&std::fs::read_to_string(base.with_extension("json")).unwrap()).unwrap()
}

/// Carga un modelo guardado por el binario y proyecta `nubes()` con él. Los
/// vectores simulados tienen 22 dimensiones, las mismas que `nubes()`.
fn proyecta(dir: &Path, nombre: &str, semilla: u64) -> Vec<Vec<f64>> {
    let modelo = FittedUmap::<MyAutodiffBackend>::load(dir.join(nombre), configuracion(semilla), 22, Default::default())
        .expect("el modelo guardado por el binario tendría que cargarse");
    modelo.transform(nubes())
}

fn bytes_distintos(a: &Path, b: &Path) -> usize {
    let (x, y) = (std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
    assert_eq!(x.len(), y.len(), "los dos .bin tendrían que medir lo mismo");
    x.iter().zip(&y).filter(|(p, q)| p != q).count()
}

#[test]
fn el_binario_misma_semilla_mismos_pesos() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("semilla");
    std::fs::create_dir_all(&dir).unwrap();
    let ma = entrena(&dir, "a", 7);
    let _ = entrena(&dir, "b", 7);
    let mc = entrena(&dir, "c", 8);
    assert_eq!(ma["formato"], 4, "el manifiesto tiene que ser de formato 4");
    assert_eq!(ma["semilla"], 7, "el manifiesto tiene que guardar la semilla");
    assert_eq!(mc["semilla"], 8);

    // Diagnóstico, no criterio: los identificadores de parámetro que burn sortea.
    println!(
        "bytes distintos en el .bin: misma semilla {}, otra semilla {}",
        bytes_distintos(&dir.join("a.bin"), &dir.join("b.bin")),
        bytes_distintos(&dir.join("a.bin"), &dir.join("c.bin"))
    );

    let (a, b, c) = (proyecta(&dir, "a", 7), proyecta(&dir, "b", 7), proyecta(&dir, "c", 8));
    let s = escala(&a);
    let d_misma = diferencia_maxima(&a, &b);
    assert_eq!(d_misma, 0.0, "dos modelos entrenados con la semilla 7 proyectan distinto, hasta {d_misma} (escala {s})");
    let d_otra = diferencia_maxima(&a, &c);
    assert!(d_otra > 0.01 * s, "los modelos de las semillas 7 y 8 proyectan casi igual ({d_otra}, escala {s})");
}
