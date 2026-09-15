//! Normalización con una referencia **fija**, decidida una vez.
//!
//! ## El problema que resuelve
//!
//! La versión anterior calculaba el mínimo y el máximo **de cada lote** y
//! reescalaba ese lote a [0,1] contra ellos. Como el mínimo y el máximo cambian
//! de un lote al siguiente, el mismo punto del encaje caía en coordenadas
//! distintas según el lote en que llegara.
//!
//! Medido sobre 62 649 formas de onda reales, siguiendo un ancla —un único punto
//! que no se mueve del encaje— a través de doce lotes consecutivos:
//!
//! ```text
//! lote   min del lote    max del lote    ancla normalizada
//!    0   -307317.8750     99222.0703    (0.0332, 0.4567)
//!    5   -297588.4062     23842.2441    (0.0118, 0.5474)
//!   10   -388817.2188     43519.0781    (0.2198, 0.6180)
//!
//! recorrido del ancla:  x 0.2080   y 0.1613
//! epsilon de DenStream:              0.1
//! ```
//!
//! El ancla se desplaza **el doble del propio `epsilon`**. DenStream agrupa por
//! distancia con ese radio, así que los micro-clusters que contaba eran en buena
//! parte estructura de la normalización y no de los datos.
//!
//! Reprodúzcase con `cargo run --release --example sonda_de_normalizacion`.
//!
//! ## La forma del arreglo
//!
//! Es la misma que el catálogo de tridesclous ya usa en el otro frente del
//! proyecto con `signals_mads` / `signals_medians`: la referencia se mide **una
//! vez**, sobre el lote de entrenamiento, y viaja con el modelo. Así dos sesiones
//! que carguen el mismo modelo comparten sistema de coordenadas.
//!
//! Sin esto, guardar el modelo no basta: aunque los pesos fueran idénticos, cada
//! lote seguiría normalizándose contra sí mismo y las coordenadas de dos sesiones
//! no serían comparables.

use serde::{Deserialize, Serialize};

/// Referencia de normalización fija, medida una vez y guardada con el modelo.
///
/// Comparte un solo par (min, max) para todas las componentes, igual que hacía
/// la versión por lote: así el encaje no se deforma por ejes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Normalizador {
    pub min: f64,
    pub max: f64,
}

impl Normalizador {
    /// Mide la referencia sobre el encaje de entrenamiento.
    ///
    /// Devuelve `None` si no hay datos o si son todos iguales — casos en los que
    /// no existe referencia posible. Se devuelve `None` en vez de un rango
    /// inventado: un respaldo callado convertiría «no se pudo medir» en «se midió
    /// otra cosa».
    pub fn ajusta(encaje: &[Vec<f64>]) -> Option<Self> {
        if encaje.is_empty() {
            return None;
        }
        let min = encaje.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max = encaje.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !min.is_finite() || !max.is_finite() || max <= min {
            return None;
        }
        Some(Self { min, max })
    }

    /// Aplica la referencia. **No recorta**: un punto que caiga fuera del rango de
    /// entrenamiento sale fuera de [0,1], y eso es información —dice que el dato
    /// es nuevo—, no un error que haya que esconder.
    pub fn aplica(&self, datos: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let rango = self.max - self.min;
        datos
            .iter()
            .map(|fila| fila.iter().map(|&x| (x - self.min) / rango).collect())
            .collect()
    }

    /// Qué fracción de los valores cae fuera de [0,1].
    ///
    /// Con la referencia fija esto deja de ser cero, y conviene vigilarlo: si
    /// crece mucho, el modelo ya no representa los datos que llegan y toca
    /// reentrenar. Es el margen que antes no se podía ni medir, porque cada lote
    /// se reescalaba a [0,1] por construcción.
    pub fn fuera_de_rango(&self, normalizados: &[Vec<f64>]) -> f64 {
        let total: usize = normalizados.iter().map(|f| f.len()).sum();
        if total == 0 {
            return 0.0;
        }
        let fuera = normalizados
            .iter()
            .flatten()
            .filter(|&&x| !(0.0..=1.0).contains(&x))
            .count();
        fuera as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Normalización de ENTRADA
// ---------------------------------------------------------------------------

/// El épsilon de `fast_umap::utils::normalize_data`, replicado para que la
/// normalización de aquí dé exactamente lo mismo que la de allí.
const EPSILON_DESVIACION: f64 = 1e-6;

/// Referencia de normalización de la **entrada** de la red, por característica.
///
/// ## Por qué hace falta
///
/// `fast-umap` 1.6.0 entrena y evalúa con preprocesados distintos:
///
/// * `FittedUmap::finalize` llama a `normalize_data()` sobre los datos de
///   entrenamiento **antes** de `model.forward()`, y el resultado es lo que
///   devuelve `embedding()`.
/// * `FittedUmap::transform_to_tensor` llama a `model.forward()` **sin**
///   normalizar.
///
/// O sea: la red se entrena sobre datos normalizados y en inferencia recibe datos
/// crudos. Medido sobre los mismos 400 vectores de entrenamiento:
///
/// ```text
/// embedding()                        rango [  -2.7453,      3.1190]
/// transform() de esos mismos, crudo  rango [-370820.03,  62719.03]
/// transform() tras normalize_data()  rango [  -2.7453,      3.1190]   <- exacto
///
/// máxima diferencia contra embedding():
///   transform(crudo)                 3.7082e5
///   transform(normalize_data(datos)) 0.0000e0
/// ```
///
/// Es un defecto del crate, no del uso. Reprodúzcase con
/// `cargo run --release --example sonda_de_transform`.
///
/// ## Por qué no basta con llamar a `normalize_data` en cada lote
///
/// `normalize_data` calcula la media y la desviación **de los datos que recibe**.
/// Llamarla por lote volvería a meter una referencia móvil, ahora en la entrada:
/// el mismo vector daría coordenadas distintas según con quién viajara. Es el
/// mismo defecto que `Normalizador` arregla en la salida.
///
/// Por eso la referencia se mide una vez, sobre el lote de entrenamiento, y
/// viaja con el modelo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizadorDeEntrada {
    /// Media de cada característica en el lote de entrenamiento.
    pub medias: Vec<f64>,
    /// Desviación típica de cada característica, con el épsilon ya sumado.
    pub desviaciones: Vec<f64>,
}

impl NormalizadorDeEntrada {
    /// Mide medias y desviaciones por característica sobre el lote de
    /// entrenamiento, igual que hace `fast_umap::utils::normalize_data`.
    pub fn ajusta(datos: &[Vec<f64>]) -> Option<Self> {
        let n = datos.len();
        let d = datos.first()?.len();
        if n == 0 || d == 0 || datos.iter().any(|f| f.len() != d) {
            return None;
        }
        let mut medias = vec![0.0; d];
        let mut desviaciones = vec![0.0; d];
        for j in 0..d {
            let suma: f64 = datos.iter().map(|f| f[j]).sum();
            let suma_cuad: f64 = datos.iter().map(|f| f[j] * f[j]).sum();
            let media = suma / n as f64;
            // Misma fórmula que el crate: varianza poblacional, no muestral.
            let varianza = (suma_cuad / n as f64) - media * media;
            medias[j] = media;
            desviaciones[j] = varianza.max(0.0).sqrt() + EPSILON_DESVIACION;
        }
        Some(Self { medias, desviaciones })
    }

    /// Aplica la referencia de entrenamiento a datos nuevos.
    ///
    /// Devuelve `Err` si la dimensión no corresponde, en vez de recortar o
    /// rellenar: un dato de otra forma no es un dato que este modelo pueda leer.
    pub fn aplica(&self, datos: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        let d = self.medias.len();
        if let Some(mala) = datos.iter().find(|f| f.len() != d) {
            return Err(format!(
                "el modelo espera {d} características y llegó un vector de {}",
                mala.len()
            ));
        }
        Ok(datos
            .iter()
            .map(|fila| {
                fila.iter()
                    .enumerate()
                    .map(|(j, &x)| (x - self.medias[j]) / self.desviaciones[j])
                    .collect()
            })
            .collect())
    }
}
