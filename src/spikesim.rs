// Genera un conjunto de Vectores n dimensionales distribuidos bajo
// la T-student para simular conjuntos de spikes de neuronas


use std::io::{self, BufRead, Write};

const DIM: usize = 22;

// ------------------------- Generador aleatorio -------------------------
struct Rng {
    s0: u64,
    s1: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // splitmix64 para expandir una sola semilla en el estado inicial
        let mut z = seed;
        let mut splitmix = move || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        let s0 = splitmix();
        let mut s1 = splitmix();
        if s1 == 0 {
            s1 = 0xA5A5A5A5A5A5A5A5;
        }
        Rng { s0, s1 }
    }

    fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
        self.s1 = s1.rotate_left(36);
        result
    }

    /// Uniforme en (0, 1),
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits de mantisa
        let u = (bits as f64) * (1.0 / (1u64 << 53) as f64);
        if u <= 0.0 {
            1e-300
        } else if u >= 1.0 {
            1.0 - 1e-16
        } else {
            u
        }
    }

    /// Normal estándar N(0,1) mediante la transformación de Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Variable Gamma(shape, escala=1) mediante Marsaglia-Tsang (2000).
    /// Válido para cualquier shape > 0.
    fn next_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            // Truco de "boost": Gamma(shape) = Gamma(shape+1) * U^(1/shape)
            let u = self.next_f64();
            return self.next_gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = self.next_normal();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Chi-cuadrada con `df` grados de libertad (df puede ser no entero).
    fn next_chi2(&mut self, df: f64) -> f64 {
        2.0 * self.next_gamma(df / 2.0)
    }

    /// t de Student estándar (media 0, escala 1) con `df` grados de libertad.
    fn next_student_t(&mut self, df: f64) -> f64 {
        let z = self.next_normal();
        let chi2 = self.next_chi2(df);
        z / (chi2 / df).sqrt()
    }
}

// ------------------------- Modelo de mezcla -------------------------

/// Una de las distribuciones t "empalmadas" (componente de la mezcla).
#[derive(Clone)]
struct ComponenteT {
    df: f64,             // grados de libertad (>0; a menor df, colas más pesadas)
    medias: [f64; DIM],  // ubicación (mu) por dimensión
    escalas: [f64; DIM], // escala (sigma) por dimensión, debe ser > 0
    peso: f64,           // peso relativo dentro de la mezcla (no hace falta que sumen 1)
}

/// Generador de vectores de `DIM` dimensiones a partir de una mezcla de
/// distribuciones t de Student ("empalmadas"). Cada llamada a
/// `generar_vector()` produce una muestra independiente.
struct GeneradorMezclaT {
    componentes: Vec<ComponenteT>,
    pesos_acumulados: Vec<f64>, // CDF discreta para elegir componente
    rng: Rng,
}

impl GeneradorMezclaT {
    fn new(componentes: Vec<ComponenteT>, seed: u64) -> Self {
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

        GeneradorMezclaT {
            componentes,
            pesos_acumulados,
            rng: Rng::new(seed),
        }
    }

    /// Elige el índice de componente según los pesos de la mezcla.
    fn elegir_componente(&mut self) -> usize {
        let u = self.rng.next_f64();
        match self
            .pesos_acumulados
            .iter()
            .position(|&p| u <= p)
        {
            Some(i) => i,
            None => self.componentes.len() - 1,
        }
    }

    /// Genera un nuevo vector de `DIM` dimensiones. Primero elige al azar
    /// una de las distribuciones empalmadas (según su peso) y luego
    /// muestrea las `DIM` componentes t de Student de esa distribución.
    fn generar_vector(&mut self) -> [f64; DIM] {
        let idx = self.elegir_componente();
        let comp = self.componentes[idx].clone();
        let mut v = [0.0f64; DIM];
        for i in 0..DIM {
            let t = self.rng.next_student_t(comp.df);
            v[i] = comp.medias[i] + comp.escalas[i] * t;
        }
        v
    }
}
