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
            // Welford en una pasada. La forma directa —`suma_cuad/n − media²`— resta
            // dos numeros grandes y parecidos, y cuando la media domina a la
            // desviacion el resultado se pierde en el redondeo:
            //
            //     |media|/desv      formula directa      Welford        real
            //              0             9,902e-1        9,902e-1     9,902e-1
            //          1e+06             9,903e-5        9,902e-5     9,902e-5
            //          1e+11             2,000e+0        9,902e-7     9,902e-7
            //
            // La ultima fila se equivoca por un factor de dos millones. Welford
            // mantiene los terminos en el mismo orden de magnitud y no cancela.
            //
            // ⚠ Esto NO arregla nada que este roto hoy. Medido sobre las 44 formas
            //   de onda del archivo RR032, la peor razon |media|/desviacion por
            //   caracteristica es 15, y la formula directa no falla hasta ~6,7e7:
            //   hay 4,5 millones de veces de margen. El cambio quita una clase de
            //   fallo que hoy no ocurre, y cuesta lo mismo.
            let mut media = 0.0_f64;
            let mut m2 = 0.0_f64;
            for (k, fila) in datos.iter().enumerate() {
                let x = fila[j];
                let delta = x - media;
                media += delta / (k + 1) as f64;
                m2 += delta * (x - media);
            }
            // Poblacional (÷n) y no muestral (÷(n−1)): es lo que hace
            // `fast_umap::utils::normalize_data` al entrenar, y la referencia tiene
            // que reproducirlo o inferencia y entrenamiento dejan de coincidir.
            let varianza = m2 / n as f64;
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

#[cfg(test)]
mod pruebas {
    use super::*;

    /// La forma directa, la que habia antes. Se conserva SOLO para la prueba de
    /// equivalencia: es contra esto contra lo que hay que comparar.
    fn desviacion_directa(datos: &[Vec<f64>], j: usize) -> f64 {
        let n = datos.len() as f64;
        let suma: f64 = datos.iter().map(|f| f[j]).sum();
        let suma_cuad: f64 = datos.iter().map(|f| f[j] * f[j]).sum();
        let media = suma / n;
        ((suma_cuad / n) - media * media).max(0.0).sqrt() + EPSILON_DESVIACION
    }

    /// Generador reproducible sin dependencias: congruencial lineal.
    fn muestras(n: usize, d: usize, base: f64, escala: f64) -> Vec<Vec<f64>> {
        let mut e: u64 = 42;
        let mut siguiente = || {
            e = e.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((e >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        (0..n).map(|_| (0..d).map(|_| base + escala * siguiente()).collect()).collect()
    }

    #[test]
    fn welford_coincide_con_la_forma_directa_cuando_esta_bien_condicionada() {
        // El cambio no puede alterar resultados en el regimen en que se venia
        // trabajando. Si los altera, no es una mejora de estabilidad: es un cambio
        // de comportamiento disfrazado.
        let datos = muestras(400, 8, 0.0, 2.0);
        let n = NormalizadorDeEntrada::ajusta(&datos).expect("debería ajustar");
        for j in 0..8 {
            let directa = desviacion_directa(&datos, j);
            assert!((n.desviaciones[j] - directa).abs() / directa < 1e-9,
                    "característica {j}: Welford {} frente a directa {directa}", n.desviaciones[j]);
        }
    }

    #[test]
    fn welford_acierta_donde_la_forma_directa_se_rompe() {
        // Media enorme y desviacion diminuta: la resta de la forma directa cancela.
        // Es el unico sitio donde el cambio se nota, y es su justificacion entera.
        //
        // Lo que devuelve cada una con estos datos (medido, no supuesto):
        //
        //     caracteristica   directa      Welford      real
        //                  0   1,414e+0     2,830e-4     2,887e-4
        //                1-3   1,000e-6     ~2,87e-4     2,887e-4
        //
        // En la 0 se equivoca por un factor de 4 900. En las otras tres la varianza
        // sale NEGATIVA, el `.max(0.0)` la recorta a cero, y lo que devuelve es el
        // epsilon: una desviacion inventada que parece un resultado.
        let datos = muestras(400, 4, 1e8, 1e-3);
        let n = NormalizadorDeEntrada::ajusta(&datos).expect("debería ajustar");
        // La dispersion real de una uniforme de anchura 1e-3 es 1e-3/sqrt(12).
        let esperada = 1e-3 / 12.0_f64.sqrt();
        for j in 0..4 {
            let welford = n.desviaciones[j];
            let directa = desviacion_directa(&datos, j);
            assert!((welford - esperada).abs() / esperada < 0.15,
                    "Welford debería acertar: {welford} frente a {esperada}");
            // Se compara por RAZON y no por error relativo: cuando la varianza se
            // recorta a cero el error relativo se queda en 0,9965 y un umbral de 1,0
            // no lo caza, aunque el resultado este mal por tres ordenes de magnitud.
            let razon = directa / welford;
            assert!(!(0.5..=2.0).contains(&razon),
                    "la forma directa ya no falla aquí (devolvió {directa} frente a \
                     {welford}): esta prueba dejó de demostrar nada y hay que buscar \
                     un caso peor");
        }
    }

    #[test]
    fn la_varianza_sigue_siendo_poblacional() {
        // Con ÷(n−1) la desviacion saldria sqrt(n/(n−1)) veces mayor, y dejaria de
        // coincidir con la que `fast_umap` aplica al entrenar. Con n=400 son 0,125 %
        // — pequeño, y exactamente la clase de divergencia silenciosa que el arreglo
        // de la normalizacion elimino.
        let datos: Vec<Vec<f64>> = vec![vec![0.0], vec![2.0]];
        let n = NormalizadorDeEntrada::ajusta(&datos).expect("debería ajustar");
        // Poblacional: media 1, varianza ((−1)² + 1²)/2 = 1, desviación 1.
        // Muestral daría sqrt(2) = 1,414.
        assert!((n.desviaciones[0] - 1.0).abs() < 1e-6,
                "desviación {} — ¿se cambió a varianza muestral?", n.desviaciones[0]);
    }

    #[test]
    fn una_caracteristica_constante_no_divide_por_cero() {
        let datos: Vec<Vec<f64>> = vec![vec![5.0], vec![5.0], vec![5.0]];
        let n = NormalizadorDeEntrada::ajusta(&datos).expect("debería ajustar");
        let z = n.aplica(vec![vec![5.0]]).expect("debería aplicar");
        assert!(z[0][0].is_finite(), "salió {}", z[0][0]);
    }
}
