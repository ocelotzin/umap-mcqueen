//! El clasificador en línea.
//!
//! Por lote hace lo mismo que `procesa_lote` en `main.rs` de umap-mcqueen:
//! normaliza la entrada con la referencia de entrenamiento, proyecta con la red,
//! normaliza el encaje con la referencia fija y alimenta el agrupador (con los
//! mismos parámetros por defecto). Lo que añade es lo que el Peeler necesita y
//! `main.rs` no hace: **una etiqueta estable por forma de onda**.

use std::error::Error;

use fast_umap::FittedUmap;
use umap_mcqueen::agrupador::{Agrupador, ParamsAgrupador};

use crate::modelo;
use crate::normaliza::{Normalizador, NormalizadorDeEntrada};
use crate::{configuracion_de, lee_manifiesto, MyAutodiffBackend};

/// -1: igual que `LABEL_TRASH` de tridesclous.
pub use umap_mcqueen::agrupador::ETIQUETA_RUIDO;

#[derive(Debug, Clone, Default)]
pub struct Resultado {
    /// Una por forma de onda, en el mismo orden. `ETIQUETA_RUIDO` si no tiene unidad.
    pub etiquetas: Vec<i64>,
    /// Coordenadas normalizadas del encaje, por si la interfaz quiere pintarlas.
    pub encaje: Vec<Vec<f32>>,
    /// Fracción del lote fuera de [0,1]: si crece, el modelo ya no representa lo que llega.
    pub fuera_de_rango: f64,
    /// Ajustes de HDBSCAN hechos hasta ahora.
    pub n_ajustes: u64,
    pub n_unidades: usize,
}

pub struct Motor {
    fitted: FittedUmap<MyAutodiffBackend>,
    norm: Normalizador,
    norm_entrada: NormalizadorDeEntrada,
    agrupador: Agrupador,
    pub n_dim: usize,
    /// Resumen legible del modelo cargado, para los registros.
    pub info: String,
}

impl Motor {
    /// Carga un modelo guardado con `--modelo`. `n_dim` es la dimensión de las
    /// formas de onda que se le van a dar; `carga_verificada` la compara con la
    /// del entrenamiento, así que un extractor mal configurado falla aquí y no
    /// en mitad de la sesión.
    pub fn carga(base: &str, n_dim: usize, params: ParamsAgrupador) -> Result<Self, Box<dyn Error>> {
        params.valida()?;
        // La configuración sale del propio manifiesto: no hay una segunda copia
        // de `configuracion()` que pueda divergir de la de umap-mcqueen.
        let config = configuracion_de(&lee_manifiesto(base)?)?;
        let (fitted, manifiesto) =
            modelo::carga_verificada::<MyAutodiffBackend>(base, config, n_dim, Default::default())?;
        let norm_entrada = manifiesto
            .normalizacion_de_entrada
            .clone()
            .ok_or("el manifiesto no trae referencia de normalización de ENTRADA")?;
        let norm = manifiesto
            .normalizacion
            .ok_or("el manifiesto no trae referencia de normalización del encaje")?;
        let info = format!(
            "{base}: entrada {n_dim}, capas {:?}, salida {}, entrenado con {} formas (huella {}…)",
            manifiesto.hidden_sizes,
            manifiesto.n_components,
            manifiesto.n_entrenamiento,
            &manifiesto.huella_datos[..manifiesto.huella_datos.len().min(12)],
        );
        let agrupador = Agrupador::new(params)?;
        Ok(Self {
            agrupador,
            fitted,
            norm,
            norm_entrada,
            n_dim,
            info,
        })
    }

    pub fn clasifica(&mut self, lote: Vec<Vec<f64>>) -> Result<Resultado, String> {
        if lote.is_empty() {
            return Ok(Resultado::default());
        }
        if let Some(f) = lote.iter().find(|f| f.len() != self.n_dim) {
            return Err(format!("forma de onda de dimensión {} y el modelo espera {}", f.len(), self.n_dim));
        }

        // `fast-umap` NO normaliza en `transform` aunque sí al entrenar
        // (ver normaliza::NormalizadorDeEntrada).
        let entrada = self.norm_entrada.aplica(lote)?;
        let encaje = self.norm.aplica(self.fitted.transform(entrada));
        let fuera_de_rango = self.norm.fuera_de_rango(&encaje);

        let puntos: Vec<Vec<f32>> = encaje
            .iter()
            .map(|fila| fila.iter().map(|&x| x as f32).collect())
            .collect();

        // Propagación de etiquetas por vecindad: lo mismo que la CLI
        // (`umap_mcqueen::agrupador::Agrupador`).
        let etiquetas = self.agrupador.etiqueta(&puntos)?;

        Ok(Resultado {
            etiquetas,
            encaje: puntos,
            fuera_de_rango,
            n_ajustes: self.agrupador.n_ajustes(),
            n_unidades: self.agrupador.n_unidades(),
        })
    }
}
