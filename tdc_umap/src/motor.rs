//! El clasificador en línea.
//!
//! Por lote hace lo mismo que `procesa_lote` en `main.rs` de umap-mcqueen:
//! normaliza la entrada con la referencia de entrenamiento, proyecta con la red,
//! normaliza el encaje con la referencia fija y alimenta DenStream (con los
//! mismos parámetros por defecto). Lo que añade es lo que el Peeler necesita y
//! `main.rs` no hace: **una etiqueta estable por forma de onda**.

use std::error::Error;

use clump::DenStream;
use fast_umap::FittedUmap;

use umap_mcqueen::denstream::Etiquetador;
use crate::modelo;
use crate::normaliza::{Normalizador, NormalizadorDeEntrada};
use crate::{configuracion_de, lee_manifiesto, MyAutodiffBackend};

/// -1: igual que `LABEL_TRASH` de tridesclous.
pub use umap_mcqueen::denstream::ETIQUETA_RUIDO;

#[derive(Debug, Clone)]
pub struct ParamsDenStream {
    /// Radio de micro-cluster. OJO: `clump` compara la distancia **al cuadrado**
    /// contra este valor al decidir si absorbe, y el radio sin elevar al cuadrado.
    pub epsilon: f32,
    pub min_pts: usize,
    pub beta: f64,
    pub lambda: f64,
    pub mu: f64,
    /// Distancia² entre centroides para unirlos en una unidad. `None` deja el
    /// valor de `clump` (2 × epsilon), que en el encaje [0,1] une casi todo.
    pub macro_epsilon: Option<f32>,
    /// `None` deja el valor de `clump` (100 actualizaciones).
    pub poda_cada: Option<usize>,
    /// Un micro-cluster que DBSCAN deja como ruido (sin vecinos suficientes) se
    /// trata como unidad propia. Sin esto, una neurona compacta que cabe entera
    /// en un solo micro-cluster sale **toda** como ruido.
    pub micro_aislado_es_unidad: bool,
}

impl Default for ParamsDenStream {
    /// Los de la CLI, que viven en `umap_mcqueen::denstream`.
    fn default() -> Self {
        use umap_mcqueen::denstream as d;
        Self {
            epsilon: d::EPSILON,
            min_pts: d::MIN_PTS,
            beta: d::BETA,
            lambda: d::LAMBDA,
            mu: d::MU,
            macro_epsilon: Some(d::MACRO_EPSILON),
            poda_cada: None,
            micro_aislado_es_unidad: true,
        }
    }
}

impl ParamsDenStream {
    /// Avisos sobre combinaciones que desactivan parte del algoritmo.
    pub fn advertencias(&self) -> Vec<String> {
        let mut v = Vec::new();
        // Un micro-cluster nuevo nace con peso 1.0. Si beta·mu <= 1, nace ya como
        // «potencial» y la ruta de outliers no se usa nunca: no hay filtro de ruido.
        let macro_eps = self.macro_epsilon.unwrap_or(2.0 * self.epsilon);
        if macro_eps >= 0.1 {
            v.push(format!(
                "macro_epsilon = {macro_eps} es una distancia AL CUADRADO: DBSCAN une \
                 centroides a {:.2} en el encaje [0,1], y tiende a juntar todas las \
                 unidades en una. Prueba 0.04 (distancia 0.2).",
                macro_eps.sqrt()
            ));
        }
        if self.beta * self.mu <= 1.0 {
            v.push(format!(
                "beta·mu = {:.2} <= 1: cada punto nuevo nace como micro-cluster potencial, \
                 así que DenStream NO separa ruido. Sube mu (p. ej. mu=10 con beta=0.5 \
                 exige ~5 puntos cercanos antes de crear una unidad).",
                self.beta * self.mu
            ));
        }
        v
    }

    fn construye(&self) -> DenStream {
        let mut ds = DenStream::new(self.epsilon, self.min_pts)
            .with_beta(self.beta)
            .with_lambda(self.lambda)
            .with_mu(self.mu);
        if let Some(e) = self.macro_epsilon {
            ds = ds.with_macro_epsilon(e);
        }
        if let Some(t) = self.poda_cada {
            ds = ds.with_pruning_period(t);
        }
        ds
    }
}

#[derive(Debug, Clone, Default)]
pub struct Resultado {
    /// Una por forma de onda, en el mismo orden. `ETIQUETA_RUIDO` si no tiene unidad.
    pub etiquetas: Vec<i64>,
    /// Coordenadas normalizadas del encaje, por si la interfaz quiere pintarlas.
    pub encaje: Vec<Vec<f32>>,
    /// Fracción del lote fuera de [0,1]: si crece, el modelo ya no representa lo que llega.
    pub fuera_de_rango: f64,
    pub n_micro: usize,
    pub n_unidades: usize,
}

pub struct Motor {
    fitted: FittedUmap<MyAutodiffBackend>,
    norm: Normalizador,
    norm_entrada: NormalizadorDeEntrada,
    etiquetador: Etiquetador,
    pub n_dim: usize,
    /// Resumen legible del modelo cargado, para los registros.
    pub info: String,
}

impl Motor {
    /// Carga un modelo guardado con `--modelo`. `n_dim` es la dimensión de las
    /// formas de onda que se le van a dar; `carga_verificada` la compara con la
    /// del entrenamiento, así que un extractor mal configurado falla aquí y no
    /// en mitad de la sesión.
    pub fn carga(
        base: &str,
        n_dim: usize,
        params: ParamsDenStream,
        tolerancia: f32,
        olvido_lotes: u64,
    ) -> Result<Self, Box<dyn Error>> {
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
        for a in params.advertencias() {
            eprintln!("[tdc_umap] Aviso: {a}");
        }
        let info = format!(
            "{base}: entrada {n_dim}, capas {:?}, salida {}, entrenado con {} formas (huella {}…)",
            manifiesto.hidden_sizes,
            manifiesto.n_components,
            manifiesto.n_entrenamiento,
            &manifiesto.huella_datos[..manifiesto.huella_datos.len().min(12)],
        );
        let etiquetador = Etiquetador::con(
            params.construye(), tolerancia, olvido_lotes, params.micro_aislado_es_unidad);
        Ok(Self {
            etiquetador,
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

        // DenStream + agrupamiento + ids estables: lo mismo que la CLI
        // (`umap_mcqueen::denstream::Etiquetador`). Los puntos con NaN/inf
        // salen como ruido sin llegar a DenStream.
        let etiquetas = self.etiquetador.etiqueta(&puntos)?;

        Ok(Resultado {
            etiquetas,
            encaje: puntos,
            fuera_de_rango,
            n_micro: self.etiquetador.n_micro(),
            n_unidades: self.etiquetador.n_unidades(),
        })
    }
}
