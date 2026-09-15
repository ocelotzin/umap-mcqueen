//! Guardado y carga **verificada** del modelo de UMAP.
//!
//! `fast_umap` ya sabe guardar y cargar los pesos —y lo hace exacto: el viaje de
//! ida y vuelta reproduce `transform()` bit a bit—. Lo que no hace es dejar
//! constancia de **con qué configuración** se entrenaron esos pesos, y su `load()`
//! acepta sin quejarse cualquier configuración que se le pase:
//!
//! ```text
//! cargar con n_neighbors=99 y metric=Cosine  -> aceptado, nadie avisa
//! cargar con hidden_sizes=[64] en vez de [128] -> aceptado, config() declara [64]
//! ```
//!
//! En los dos casos los pesos del disco ganan, o sea que la red cargada es la
//! correcta; lo que queda mal es el `config` que la describe. Un modelo que miente
//! sobre sus propios hiperparámetros no sirve para comparar sesiones, que es
//! justamente para lo que se guarda.
//!
//! Este módulo añade un **manifiesto** junto a los pesos y una carga que **falla**
//! cuando no corresponden. No sustituye a `fast_umap::serialize`: lo envuelve.

use fast_umap::backend::AutodiffBackend;
use fast_umap::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::path::{Path, PathBuf};

/// Lo que hay que saber de un modelo guardado para poder confiar en él.
///
/// No incluye los pesos: esos los escribe `fast_umap` en el `.bin` de al lado.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifiesto {
    /// Versión del propio manifiesto, para que un cambio de formato se note.
    pub formato: u32,
    /// Dimensiones de salida del encaje.
    pub n_components: usize,
    /// Capas ocultas de la red. Determina la arquitectura: si no corresponde,
    /// los pesos del disco son de otra red.
    pub hidden_sizes: Vec<usize>,
    /// Vecinos del grafo. No toca la arquitectura, pero sí el significado.
    pub n_neighbors: usize,
    /// Métrica del grafo, como texto (el enum de `fast_umap` no es serializable).
    pub metrica: String,
    /// Dimensiones de entrada. `load()` lo exige y no lo puede adivinar.
    pub input_size: usize,
    /// Cuántas formas de onda se usaron para entrenar.
    pub n_entrenamiento: usize,
    /// SHA-256 de los datos de entrenamiento, para saber si son los mismos.
    pub huella_datos: String,
    /// Segundos desde el epoch. Sin `chrono`: no hace falta una dependencia
    /// para un sello de tiempo.
    pub guardado_en: u64,
}

/// Qué no cuadra entre lo que se pide y lo que hay guardado.
#[derive(Debug)]
pub struct Discrepancia {
    pub campo: &'static str,
    pub guardado: String,
    pub pedido: String,
}

impl std::fmt::Display for Discrepancia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: guardado {}, se pidió {}", self.campo, self.guardado, self.pedido)
    }
}

/// Nombre textual de la métrica. El enum de `fast_umap` no implementa `Serialize`,
/// así que se traduce a mano — y el `_` se deja explícito para que una métrica
/// nueva del crate no se guarde en silencio como otra cosa.
pub fn nombre_de_metrica(m: &Metric) -> String {
    match m {
        Metric::Euclidean => "Euclidean".into(),
        Metric::Manhattan => "Manhattan".into(),
        Metric::Cosine => "Cosine".into(),
        otra => format!("{otra:?}"),
    }
}

/// Huella de los datos de entrenamiento.
///
/// Se calcula sobre los bytes de los `f64` en orden, no sobre el fichero: así dos
/// CSV con el mismo contenido y distinto formato dan la misma huella, que es lo
/// que interesa saber.
pub fn huella(datos: &[Vec<f64>]) -> String {
    let mut h = Sha256::new();
    for fila in datos {
        for x in fila {
            h.update(x.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

impl Manifiesto {
    /// Construye el manifiesto que describe un entrenamiento concreto.
    pub fn de(config: &UmapConfig, datos: &[Vec<f64>]) -> Self {
        Self {
            formato: 1,
            n_components: config.n_components,
            hidden_sizes: config.hidden_sizes.clone(),
            n_neighbors: config.graph.n_neighbors,
            metrica: nombre_de_metrica(&config.graph.metric),
            input_size: datos.first().map(|f| f.len()).unwrap_or(0),
            n_entrenamiento: datos.len(),
            huella_datos: huella(datos),
            guardado_en: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Todo lo que no cuadra entre este manifiesto y una configuración pedida.
    ///
    /// Devuelve la lista completa, no el primer fallo: si hay tres cosas mal,
    /// conviene verlas de una vez y no una por corrida.
    pub fn discrepancias(&self, config: &UmapConfig, input_size: usize) -> Vec<Discrepancia> {
        let mut v = Vec::new();
        let mut comprueba = |campo, guardado: String, pedido: String| {
            if guardado != pedido {
                v.push(Discrepancia { campo, guardado, pedido });
            }
        };
        comprueba("n_components", self.n_components.to_string(), config.n_components.to_string());
        comprueba("hidden_sizes", format!("{:?}", self.hidden_sizes), format!("{:?}", config.hidden_sizes));
        comprueba("n_neighbors", self.n_neighbors.to_string(), config.graph.n_neighbors.to_string());
        comprueba("metrica", self.metrica.clone(), nombre_de_metrica(&config.graph.metric));
        comprueba("input_size", self.input_size.to_string(), input_size.to_string());
        v
    }
}

/// Las dos rutas que componen un modelo guardado.
///
/// Existe porque `fast_umap::save` **no escribe en la ruta que se le pasa**: el
/// `BinFileRecorder` de `burn` sustituye la extensión por `.bin`. Pedir
/// `modelo.umap` deja en disco `modelo.bin`, y un `Path::exists("modelo.umap")`
/// posterior devuelve `false` sobre un modelo que sí se guardó.
#[derive(Debug, Clone)]
pub struct RutasDelModelo {
    /// Donde `burn` deja los pesos, de verdad.
    pub pesos: PathBuf,
    /// Donde este módulo deja el manifiesto.
    pub manifiesto: PathBuf,
    /// Lo que hay que pasarle a `fast_umap`, sin extensión.
    base: PathBuf,
}

impl RutasDelModelo {
    /// A partir de una ruta base (con extensión o sin ella).
    pub fn de(base: impl AsRef<Path>) -> Self {
        let base = base.as_ref().with_extension("");
        Self {
            pesos: base.with_extension("bin"),
            manifiesto: base.with_extension("json"),
            base,
        }
    }
}

/// Guarda pesos y manifiesto, y devuelve dónde quedaron de verdad.
pub fn guarda<B: AutodiffBackend>(
    ajustado: &FittedUmap<B>,
    config: &UmapConfig,
    datos_de_entrenamiento: &[Vec<f64>],
    base: impl AsRef<Path>,
) -> Result<RutasDelModelo, Box<dyn Error>> {
    let rutas = RutasDelModelo::de(base);
    if let Some(dir) = rutas.pesos.parent() {
        std::fs::create_dir_all(dir)?;
    }
    ajustado.save(&rutas.base)?;
    if !rutas.pesos.exists() {
        return Err(format!(
            "fast_umap dijo que guardó pero no hay nada en {}",
            rutas.pesos.display()
        )
        .into());
    }
    let manifiesto = Manifiesto::de(config, datos_de_entrenamiento);
    std::fs::write(&rutas.manifiesto, serde_json::to_string_pretty(&manifiesto)?)?;
    Ok(rutas)
}

/// Carga un modelo **comprobando** que el manifiesto corresponde a lo que se pide.
///
/// Falla —en vez de devolver un modelo que se describe mal— si falta el
/// manifiesto o si algún campo no cuadra.
pub fn carga_verificada<B: AutodiffBackend>(
    base: impl AsRef<Path>,
    config: UmapConfig,
    input_size: usize,
    device: burn::tensor::Device<B>,
) -> Result<(FittedUmap<B>, Manifiesto), Box<dyn Error>> {
    let rutas = RutasDelModelo::de(base);
    if !rutas.pesos.exists() {
        return Err(format!("no hay pesos en {}", rutas.pesos.display()).into());
    }
    if !rutas.manifiesto.exists() {
        return Err(format!(
            "hay pesos en {} pero no su manifiesto en {}. Sin manifiesto no se puede \
             comprobar que la configuración corresponde, y `fast_umap::load` acepta \
             cualquiera sin avisar: cargar así daría un modelo que miente sobre sí mismo.",
            rutas.pesos.display(),
            rutas.manifiesto.display()
        )
        .into());
    }

    let manifiesto: Manifiesto = serde_json::from_str(&std::fs::read_to_string(&rutas.manifiesto)?)?;
    let malas = manifiesto.discrepancias(&config, input_size);
    if !malas.is_empty() {
        let detalle: Vec<String> = malas.iter().map(|d| format!("  - {d}")).collect();
        return Err(format!(
            "el modelo guardado no corresponde a la configuración pedida:\n{}",
            detalle.join("\n")
        )
        .into());
    }

    let modelo = FittedUmap::<B>::load(&rutas.base, config, input_size, device)?;
    Ok((modelo, manifiesto))
}
