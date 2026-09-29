//! Módulo para generar vectores de `DIM` dimensiones a partir de una
//! MEZCLA de distribuciones t de Student ("distribuciones empalmadas").
//!
//! Uso típico:
//! ```ignore
//! mod spikesim;
//! use spikesim::{ComponenteT, GeneradorMezclaT};
//!
//! // Componentes definidos a mano:
//! let mut gen = GeneradorMezclaT::new(vec![
//!     ComponenteT::simetrica(3.0, 0.0, 1.0, 0.5),
//! ]);
//!
//! // O con n distribuciones de parámetros aleatorios, reproducibles vía semilla:
//! let mut gen2 = GeneradorMezclaT::aleatorio(5, 42);
//! let v = gen2.generar_vector();
//! ```

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng};
use rand_distr::{Distribution, StudentT};

pub const DIM: usize = 22;

// Rangos por defecto usados al generar componentes con parámetros
// aleatorios (ajustable)
const DF_MIN: f64 = 2.0;
const DF_MAX: f64 = 30.0;
const MEDIA_MIN: f64 = -5.0;
const MEDIA_MAX: f64 = 5.0;
const ESCALA_MIN: f64 = 0.1;
const ESCALA_MAX: f64 = 3.0;
const PESO_MIN: f64 = 0.1;
const PESO_MAX: f64 = 1.0;

/// Una de las distribuciones t "empalmadas" (componente de la mezcla).
#[derive(Clone, Debug)]
pub struct ComponenteT {
    pub df: f64,             // grados de libertad (>0; a menor df, colas más pesadas)
    pub medias: [f64; DIM],  // ubicación (mu) por dimensión
    pub escalas: [f64; DIM], // escala (sigma) por dimensión, debe ser > 0
    pub peso: f64,           // peso relativo en la mezcla (no hace falta que sumen 1)
}

#[allow(dead_code)] //// AAAAAAA debería no usar esto, pero quiero tener todas las 
                    /// funciones disponibles
impl ComponenteT {
    /// Control total: media y escala distintas por dimensión.
    pub fn nueva(df: f64, medias: [f64; DIM], escalas: [f64; DIM], peso: f64) -> Self {
        Self { df, medias, escalas, peso }
    }

    /// Atajo para el caso común: la misma media y escala en las 22 dimensiones.
    pub fn simetrica(df: f64, media: f64, escala: f64, peso: f64) -> Self {
        Self { df, medias: [media; DIM], escalas: [escala; DIM], peso }
    }

    /// t-Student estándar (media 0, escala 1, peso 1).
    pub fn estandar(df: f64) -> Self {
        Self::simetrica(df, 0.0, 1.0, 1.0)
    }

    /// Componente con parámetros aleatorios, muestreados con `rng`
    /// dentro de los rangos por defecto del módulo.
    fn aleatoria(rng: &mut StdRng) -> Self {
        let dist_df = Uniform::new(DF_MIN, DF_MAX);
        let dist_media = Uniform::new(MEDIA_MIN, MEDIA_MAX);
        let dist_escala = Uniform::new(ESCALA_MIN, ESCALA_MAX);
        let dist_peso = Uniform::new(PESO_MIN, PESO_MAX);

        let df = rng.sample(dist_df);
        let peso = rng.sample(dist_peso);
        let mut medias = [0.0; DIM];
        let mut escalas = [0.0; DIM];
        for i in 0..DIM {
            medias[i] = rng.sample(dist_media);
            escalas[i] = rng.sample(dist_escala);
        }
        Self::nueva(df, medias, escalas, peso)
    }
}

/// Generador de vectores de `DIM` dimensiones a partir de una mezcla de
/// distribuciones t de Student. Implementa `Iterator`, así que puedes
/// tratarlo como una fuente infinita de vectores aleatorios.
pub struct GeneradorMezclaT {
    componentes: Vec<ComponenteT>,
    pesos_acumulados: Vec<f64>,
    rng: StdRng,
}

#[allow(dead_code)] /// Igual no quiero pero es mejor asi
impl GeneradorMezclaT {
    /// Construye el generador a partir de componentes ya definidos.
    /// La semilla interna sale de la entropía del sistema (no reproducible).
    pub fn new(componentes: Vec<ComponenteT>) -> Self {
        Self::construir(componentes, StdRng::from_entropy())
    }

    /// Igual que `new`, pero con una semilla explícita: dos generadores
    /// creados con los mismos componentes y la misma semilla producen
    /// exactamente la misma secuencia de vectores.
    pub fn new_con_semilla(componentes: Vec<ComponenteT>, semilla: u64) -> Self {
        Self::construir(componentes, StdRng::seed_from_u64(semilla))
    }

    /// Crea el generador con `n` distribuciones t "empalmadas", cada una
    /// con parámetros (df, medias, escalas, peso) elegidos al azar. Toda
    /// la aleatoriedad —tanto la de los parámetros como la de los
    /// vectores que genere después— sale de una sola semilla, así que
    /// el resultado es 100% reproducible: misma `n` + misma `semilla`
    /// => mismas distribuciones y mismos vectores generados después.
    pub fn aleatorio(n: usize, semilla: u64) -> Self {
        assert!(n > 0, "n debe ser al menos 1");
        let mut rng = StdRng::seed_from_u64(semilla);
        let componentes: Vec<ComponenteT> = (0..n).map(|_| ComponenteT::aleatoria(&mut rng)).collect();
        Self::construir(componentes, rng)
    }

    fn construir(componentes: Vec<ComponenteT>, rng: StdRng) -> Self {
        assert!(!componentes.is_empty(), "Debe haber al menos una distribución");
        let total: f64 = componentes.iter().map(|c| c.peso).sum();
        assert!(total > 0.0, "La suma de los pesos debe ser positiva");

        let mut acumulado = 0.0;
        let pesos_acumulados: Vec<f64> = componentes
            .iter()
            .map(|c| {
                acumulado += c.peso / total;
                acumulado
            })
            .collect();

        Self { componentes, pesos_acumulados, rng }
    }

    /// Muestra de solo lectura de los componentes actuales (útil para
    /// saber qué distribuciones quedaron al usar `aleatorio`).
    pub fn componentes(&self) -> &[ComponenteT] {
        &self.componentes
    }

    fn elegir_componente(&mut self) -> usize {
        let u: f64 = self.rng.r#gen();
        self.pesos_acumulados
            .iter()
            .position(|&p| u <= p)
            .unwrap_or(self.componentes.len() - 1)
    }

    /// Genera UN vector nuevo. Cada llamada es independiente: elige al
    /// azar una de las distribuciones empalmadas (según su peso) y
    /// muestrea las `DIM` componentes t de esa distribución.
    pub fn generar_vector(&mut self) -> [f64; DIM] {
        let idx = self.elegir_componente();
        let comp = self.componentes[idx].clone();
        let dist = StudentT::new(comp.df).expect("df debe ser > 0");

        let mut v = [0.0f64; DIM];
        for i in 0..DIM {
            v[i] = comp.medias[i] + comp.escalas[i] * dist.sample(&mut self.rng);
        }
        v
    }

    /// Atajo para generar `n` vectores de una vez.
    pub fn generar_vectores(&mut self, n: usize) -> Vec<[f64; DIM]> {
        (0..n).map(|_| self.generar_vector()).collect()
    }
}

/// Permite usar el generador como un iterador infinito:
/// `generador.by_ref().take(10).collect::<Vec<_>>()`
impl Iterator for GeneradorMezclaT {
    type Item = [f64; DIM];
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generar_vector())
    }
}
