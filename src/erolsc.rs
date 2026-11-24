//! **EROLSC** — *Extended Robust On-Line Streaming Clustering* (Ibrahim, Du y
//! Keller, IPMU 2018, CCIS 853, pp. 467–478), aplicado sobre el encaje UMAP con
//! una **ventana movible** de `ventana` puntos (por defecto 4000).
//!
//! # El algoritmo del artículo (Fig. 1)
//!
//! **Inicialización**, sobre los primeros `S` puntos:
//! 1. PCM (*Possibilistic C-Means*) con `C` prototipos. Con `C` mayor que el
//!    número real de grupos, varios prototipos convergen a la misma moda
//!    (clusters coincidentes) y se fusionan: es el papel de AMPCM.
//! 2. Los puntos cuya pertenencia a **todos** los grupos es menor que `Tn` van
//!    al historial de anomalías.
//! 3. Cada grupo con más de `M` puntos se vuelve una gaussiana (media,
//!    covarianza, cuenta). Los grupos pequeños van al historial de anomalías.
//!
//! **Actualización**, punto a punto:
//! 1. Distancia de Mahalanobis (ec. 1) a todas las gaussianas.
//! 2. Si la mínima es menor que `Td`, el punto entra a esa gaussiana y se
//!    actualizan su media y su covarianza (ec. 2 y 3). Si no, es anomalía.
//!
//! **Revisión del historial de anomalías**:
//! 1. Las anomalías que ya caen a menos de `Td` de alguna gaussiana (porque
//!    ésta creció hacia ellas) entran a esa gaussiana.
//! 2. PCM con `C ≥ 10` sobre las anomalías restantes. Cada grupo **legítimo**
//!    con más de `M` puntos nace como gaussiana nueva. Así se detectan
//!    **varias** estructuras emergentes a la vez, que es la extensión de EROLSC
//!    sobre ROLSC.
//!
//! # La ventana movible
//!
//! El artículo olvida cada punto en cuanto actualiza la gaussiana: las
//! gaussianas acumulan toda la historia. Aquí las estadísticas de cada
//! gaussiana se llevan **sólo sobre los últimos `ventana` puntos** del encaje:
//!
//! - cada punto que entra suma a su gaussiana; el más viejo de la ventana
//!   **resta** de la suya al salir. La media y la covarianza son exactamente
//!   las de los puntos de esa gaussiana dentro de la ventana, así que siguen a
//!   una neurona que deriva en vez de quedarse con su posición promedio de
//!   toda la sesión;
//! - el historial de anomalías son los puntos de la ventana sin grupo: una
//!   anomalía que sale de la ventana sin haber entrado a ninguna gaussiana se
//!   queda como anomalía para siempre. Así el historial no crece sin límite;
//! - si una gaussiana se queda con menos de `M` puntos en la ventana (su
//!   neurona calló), **conserva** sus últimos parámetros en vez de estimarlos
//!   con dos o tres puntos: una covarianza degenerada la volvería incapaz de
//!   absorber nada y la mataría sin que nadie lo decidiera.
//!
//! La revisión de anomalías se hace al final de cada lote, no tras cada punto:
//! un PCM por espiga sería caro y no cambia lo que se detecta.
//!
//! # Decisiones que el artículo deja abiertas
//!
//! - **AMPCM** (Yang y Lai, 2011) se reduce a su efecto: tras PCM, dos
//!   prototipos se fusionan si cada uno tiene pertenencia ≥ ½ en el otro
//!   (`‖vᵢ − vⱼ‖² < min(ηᵢ, ηⱼ)`).
//! - **PCM** (Krishnapuram y Keller, 1993) con `m = 2`, inicializado con FCM, y
//!   `ηᵢ` = media de `d²` ponderada por `u²` (K = 1).
//! - **Legitimidad** de un grupo emergente: su dispersión (raíz de la varianza
//!   media) no pasa de `factor_dispersion` × la mayor dispersión entre las
//!   gaussianas existentes. Una nube difusa de ruido no nace como unidad.

use std::collections::VecDeque;

/// Etiqueta de un punto que no pertenece a ninguna gaussiana: anomalía.
pub const RUIDO: i64 = -1;
/// Etiqueta de un punto que llegó antes de que la ventana inicial se llenara y
/// que salió de ella antes de la inicialización (sólo con lotes mayores que la
/// ventana), o que se emitió antes de inicializar.
pub const SIN_CLASIFICAR: i64 = -2;

/// Tamaño por defecto de la ventana movible, `S` en el artículo.
pub const VENTANA: usize = 4000;

#[derive(Clone, Debug)]
pub struct Parametros {
    /// `S`: puntos de la inicialización y largo de la ventana movible.
    pub ventana: usize,
    /// `M`: puntos mínimos para que un grupo sea gaussiana.
    pub min_puntos: usize,
    /// `Td`: umbral de distancia de Mahalanobis. Con 2 dimensiones, 2.25 deja
    /// fuera ≈ 8 % de una gaussiana (χ²₂ = 5.06).
    pub umbral_distancia: f64,
    /// `Tn`: pertenencia posibilística mínima a algún grupo para no ser anomalía.
    pub umbral_pertenencia: f64,
    /// `C` de PCM en la inicialización.
    pub c_inicial: usize,
    /// `C` de PCM sobre el historial de anomalías (el artículo pide `C ≥ 10`).
    pub c_anomalias: usize,
    /// Legitimidad de un grupo emergente (ver la documentación del módulo).
    pub factor_dispersion: f64,
    /// Fusión de prototipos de PCM: se unen si `‖vᵢ − vⱼ‖² < factor_fusion · min(ηᵢ, ηⱼ)`.
    pub factor_fusion: f64,
    /// Semilla del sembrado de PCM, para que una corrida se pueda repetir.
    pub semilla: u64,
}

/// Valores calibrados con `examples/sonda_de_erolsc.rs` sobre los encajes de
/// los 16 canales de `monocanal` (híbrido de Janelia con deriva, 5–11 unidades
/// por canal; encaje de UMAP con semilla 9999, el mismo que usa la rama dev),
/// contra las etiquetas verdaderas, con lotes de 200. ARI contando las
/// anomalías de EROLSC como una clase más; «validación cruzada» = cada canal
/// evaluado con los `Td`, `Tn` y `M` que mejor ARI medio dan en los otros 15.
///
/// | Ventana | Td   | Tn  | M   | ARI medio | Validación cruzada | Anomalías |
/// |---------|------|-----|-----|-----------|--------------------|-----------|
/// | 1000    | 3.0  | 0.1 | 20  | 0.127     | —                  | 3 %       |
/// | 1000    | 2.5  | 0.5 | 50  | 0.252     | 0.188–0.234        | 8 %       |
/// | 2000    | 2.5  | 0.6 | 50  | 0.305     | 0.279              | 9 %       |
/// | 4000    | 2.25 | 0.5 | 200 | 0.315     | 0.302              | 13 %      |
/// | 8000    | 2.5  | 0.6 | 200 | 0.295     | 0.295              | 13 %      |
///
/// La primera fila son los valores de partida; las demás, lo mejor de cada
/// ventana (para 1000, la validación cruzada da 0.188 con la rejilla amplia y
/// 0.234 con la reducida alrededor del óptimo). Con la ventana 4000 la
/// validación cruzada elige la misma combinación en 14 de 16 canales, y `M`
/// queda en el interior de la rejilla (100: 0.304, 400: 0.280, 800: 0.178).
/// En los mismos canales y el mismo encaje, el agrupador de `dev` da ARI medio
/// 0.171. Agrupar un canal entero (~50 000 espigas) tarda menos de un segundo
/// con 4000. Todo sale de un solo registro: falta confirmarlo en otro.
impl Default for Parametros {
    fn default() -> Self {
        Parametros {
            ventana: VENTANA,
            min_puntos: 200,
            umbral_distancia: 2.25,
            umbral_pertenencia: 0.5,
            c_inicial: 10,
            c_anomalias: 10,
            factor_dispersion: 2.0,
            factor_fusion: 1.0,
            semilla: 0x5EED,
        }
    }
}

impl Parametros {
    pub fn valida(&self) -> Result<(), String> {
        if self.ventana == 0 {
            return Err("la ventana debe tener al menos un punto".into());
        }
        if self.min_puntos == 0 || self.min_puntos >= self.ventana {
            return Err(format!(
                "min_puntos ({}) debe estar entre 1 y la ventana ({})",
                self.min_puntos, self.ventana
            ));
        }
        if !(self.umbral_distancia > 0.0) {
            return Err("umbral_distancia debe ser positivo".into());
        }
        if !(self.umbral_pertenencia > 0.0 && self.umbral_pertenencia < 1.0) {
            return Err("umbral_pertenencia debe estar en (0, 1)".into());
        }
        if self.c_inicial == 0 || self.c_anomalias == 0 {
            return Err("C de PCM debe ser al menos 1".into());
        }
        if !(self.factor_fusion >= 0.0) {
            return Err("factor_fusion no puede ser negativo".into());
        }
        if !(self.factor_dispersion > 0.0) {
            return Err("factor_dispersion debe ser positivo".into());
        }
        Ok(())
    }
}

/// Una componente gaussiana, con sus estadísticas sobre la ventana.
#[derive(Clone, Debug)]
struct Gaussiana {
    id: i64,
    /// Puntos de esta gaussiana dentro de la ventana.
    n: usize,
    suma: Vec<f64>,
    /// Σ x xᵀ, por filas.
    suma_cuad: Vec<f64>,
    media: Vec<f64>,
    /// Factor de Cholesky (triangular inferior, por filas) de la covarianza.
    chol: Vec<f64>,
    /// Raíz de la varianza media: la dispersión del grupo.
    dispersion: f64,
}

impl Gaussiana {
    /// `None` si los puntos no alcanzan para una covarianza.
    fn de_puntos(id: i64, puntos: &[&[f64]]) -> Option<Self> {
        let d = puntos.first()?.len();
        let mut g = Gaussiana {
            id,
            n: 0,
            suma: vec![0.0; d],
            suma_cuad: vec![0.0; d * d],
            media: vec![0.0; d],
            chol: vec![0.0; d * d],
            dispersion: 0.0,
        };
        for x in puntos {
            g.acumula(x, 1.0);
        }
        g.estima(1, 1.0).then_some(g)
    }

    fn acumula(&mut self, x: &[f64], signo: f64) {
        let d = x.len();
        for i in 0..d {
            self.suma[i] += signo * x[i];
            for j in 0..d {
                self.suma_cuad[i * d + j] += signo * x[i] * x[j];
            }
        }
        if signo > 0.0 {
            self.n += 1;
        } else {
            self.n -= 1;
        }
    }

    /// Recalcula media y covarianza desde las sumas de la ventana, con la
    /// covarianza multiplicada por `correccion` (ver [`correccion_truncado`]).
    /// Con menos de `min_puntos` puntos, o si la covarianza no es definida
    /// positiva, se conservan los parámetros anteriores y devuelve `false`.
    fn estima(&mut self, min_puntos: usize, correccion: f64) -> bool {
        let d = self.media.len();
        if self.n < min_puntos.max(2) {
            return false;
        }
        let n = self.n as f64;
        let media: Vec<f64> = self.suma.iter().map(|s| s / n).collect();
        let mut cov = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                cov[i * d + j] = correccion * (self.suma_cuad[i * d + j] / n - media[i] * media[j]);
            }
        }
        let traza: f64 = (0..d).map(|i| cov[i * d + i]).sum();
        //Regularización mínima: un grupo casi colineal en el encaje no debe
        //dar una covarianza singular.
        let reg = 1e-12 + 1e-6 * traza.max(0.0) / d as f64;
        for i in 0..d {
            cov[i * d + i] += reg;
        }
        let Some(chol) = cholesky(&cov, d) else { return false };
        self.media = media;
        self.chol = chol;
        self.dispersion = (traza.max(0.0) / d as f64).sqrt();
        true
    }

    /// Ec. 1: √((x − μ)ᵀ Σ⁻¹ (x − μ)), resolviendo L y = x − μ.
    fn mahalanobis(&self, x: &[f64]) -> f64 {
        let d = self.media.len();
        let mut y = vec![0.0; d];
        let mut q = 0.0;
        for i in 0..d {
            let mut s = x[i] - self.media[i];
            for k in 0..i {
                s -= self.chol[i * d + k] * y[k];
            }
            y[i] = s / self.chol[i * d + i];
            q += y[i] * y[i];
        }
        q.sqrt()
    }
}

/// Factor por el que hay que multiplicar la covarianza muestral de los puntos
/// de una gaussiana de dimensión `d` que cayeron a distancia de Mahalanobis
/// menor que `td`, para recuperar la covarianza de la gaussiana completa:
/// `P(χ²_d < td²) / P(χ²_{d+2} < td²)`, la corrección de consistencia del MCD.
///
/// Sin ella cada gaussiana se estima sólo con lo que absorbió, sale más
/// estrecha que la nube real, y con la ventana movible se encoge un poco más en
/// cada renovación de la ventana (≈ 14 % de varianza por vuelta con `Td` 2.5):
/// las colas quedan como anomalías y acaban naciendo como grupos falsos
/// alrededor del verdadero.
fn correccion_truncado(d: usize, td: f64) -> f64 {
    let x = td * td / 2.0;
    let dentro = gamma_inferior_regularizada(d as f64 / 2.0, x);
    let dentro_mas_2 = gamma_inferior_regularizada(d as f64 / 2.0 + 1.0, x);
    if dentro_mas_2 > 0.0 { dentro / dentro_mas_2 } else { 1.0 }
}

/// P(a, x) por su serie; `a` es entero o semientero (la mitad de una dimensión).
fn gamma_inferior_regularizada(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    //Γ(a + 1) por recurrencia desde Γ(1) = 1 o Γ(½) = √π.
    let (mut g, mut z) = if (a.fract() - 0.5).abs() < 1e-9 { (std::f64::consts::PI.sqrt(), 0.5) } else { (1.0, 1.0) };
    while z < a + 1.0 - 1e-9 {
        g *= z;
        z += 1.0;
    }
    let mut termino = 1.0;
    let mut suma = 1.0;
    for k in 1..500 {
        termino *= x / (a + k as f64);
        suma += termino;
        if termino < 1e-16 * suma {
            break;
        }
    }
    (suma * (-x + a * x.ln()).exp() / g).min(1.0)
}

fn cholesky(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

/// Un punto de la ventana con su etiqueta actual.
#[derive(Clone, Debug)]
struct Entrada {
    t: u64,
    x: Vec<f64>,
    etiqueta: i64,
}

/// Agrupador EROLSC con ventana movible.
pub struct Erolsc {
    p: Parametros,
    dim: Option<usize>,
    ventana: VecDeque<Entrada>,
    gaussianas: Vec<Gaussiana>,
    inicializado: bool,
    siguiente_id: i64,
    t: u64,
    rng: Rng,
    /// Corrección de la covarianza por truncado en `Td`; se fija con la dimensión.
    correccion: f64,
}

impl Erolsc {
    pub fn new(p: Parametros) -> Result<Self, String> {
        p.valida()?;
        Ok(Erolsc {
            dim: None,
            ventana: VecDeque::with_capacity(p.ventana + 1),
            gaussianas: Vec::new(),
            inicializado: false,
            siguiente_id: 0,
            t: 0,
            rng: Rng::new(p.semilla),
            correccion: 1.0,
            p,
        })
    }

    pub fn params(&self) -> &Parametros {
        &self.p
    }

    /// Procesa un lote del encaje y devuelve la etiqueta de cada punto: el id de
    /// su gaussiana, [`RUIDO`] o, antes de que se llene la ventana inicial,
    /// [`SIN_CLASIFICAR`].
    ///
    /// La etiqueta es la que el punto tiene **al terminar el lote**: una
    /// anomalía que la revisión del lote absorbe o vuelve grupo nuevo sale ya
    /// con su grupo.
    pub fn actualiza_lote(&mut self, puntos: &[Vec<f64>]) -> Result<Vec<i64>, String> {
        let t0 = self.t;
        let mut salida = vec![SIN_CLASIFICAR; puntos.len()];
        for x in puntos {
            let d = match self.dim {
                Some(d) => d,
                None => {
                    self.correccion = correccion_truncado(x.len(), self.p.umbral_distancia);
                    *self.dim.insert(x.len())
                }
            };
            if x.len() != d {
                return Err(format!("punto de dimensión {} en un encaje de dimensión {d}", x.len()));
            }
            if x.iter().any(|v| !v.is_finite()) {
                return Err("punto con coordenadas no finitas".into());
            }
            self.empuja(x.clone(), t0, &mut salida);
            if !self.inicializado && self.ventana.len() == self.p.ventana {
                self.inicializa();
            }
        }
        if self.inicializado {
            self.revisa_anomalias();
        }
        for e in &self.ventana {
            if e.t >= t0 {
                salida[(e.t - t0) as usize] = if self.inicializado { e.etiqueta } else { SIN_CLASIFICAR };
            }
        }
        Ok(salida)
    }

    /// Mete un punto a la ventana: saca el más viejo (restándolo de su
    /// gaussiana) y, si ya hay modelo, clasifica el nuevo.
    fn empuja(&mut self, x: Vec<f64>, t0: u64, salida: &mut [i64]) {
        if self.ventana.len() == self.p.ventana {
            let viejo = self.ventana.pop_front().expect("ventana llena");
            if viejo.t >= t0 {
                salida[(viejo.t - t0) as usize] = if self.inicializado { viejo.etiqueta } else { SIN_CLASIFICAR };
            }
            if viejo.etiqueta >= 0 {
                let (min_puntos, corr) = (self.p.min_puntos, self.correccion);
                if let Some(g) = self.gaussianas.iter_mut().find(|g| g.id == viejo.etiqueta) {
                    g.acumula(&viejo.x, -1.0);
                    g.estima(min_puntos, corr);
                }
            }
        }
        let etiqueta = if self.inicializado { self.asigna(&x) } else { SIN_CLASIFICAR };
        self.ventana.push_back(Entrada { t: self.t, x, etiqueta });
        self.t += 1;
    }

    /// Ec. 1–3: la gaussiana más cercana por Mahalanobis, si está a menos de `Td`.
    fn asigna(&mut self, x: &[f64]) -> i64 {
        let Some((k, dist)) = self.mas_cercana(x) else { return RUIDO };
        if dist >= self.p.umbral_distancia {
            return RUIDO;
        }
        let g = &mut self.gaussianas[k];
        g.acumula(x, 1.0);
        g.estima(self.p.min_puntos, self.correccion);
        g.id
    }

    fn mas_cercana(&self, x: &[f64]) -> Option<(usize, f64)> {
        self.gaussianas
            .iter()
            .enumerate()
            .map(|(k, g)| (k, g.mahalanobis(x)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn nuevo_id(&mut self) -> i64 {
        let id = self.siguiente_id;
        self.siguiente_id += 1;
        id
    }

    /// PCM sobre la ventana inicial; cada grupo con más de `M` puntos es una
    /// gaussiana, el resto es anomalía.
    fn inicializa(&mut self) {
        let idx: Vec<usize> = (0..self.ventana.len()).collect();
        self.siembra_grupos(&idx, self.p.c_inicial, false);
        for e in self.ventana.iter_mut() {
            if e.etiqueta == SIN_CLASIFICAR {
                e.etiqueta = RUIDO;
            }
        }
        self.inicializado = true;
    }

    /// Revisión del historial de anomalías (las entradas [`RUIDO`] de la ventana).
    fn revisa_anomalias(&mut self) {
        //1. Anomalías a las que una gaussiana ya llegó.
        for i in 0..self.ventana.len() {
            if self.ventana[i].etiqueta != RUIDO {
                continue;
            }
            let x = self.ventana[i].x.clone();
            let id = self.asigna(&x);
            self.ventana[i].etiqueta = id;
        }
        //2. Estructuras emergentes entre las que quedan.
        let anomalas: Vec<usize> = (0..self.ventana.len())
            .filter(|&i| self.ventana[i].etiqueta == RUIDO)
            .collect();
        if anomalas.len() > self.p.min_puntos {
            self.siembra_grupos(&anomalas, self.p.c_anomalias, true);
        }
    }

    /// PCM sobre las entradas `idx` de la ventana; cada grupo con más de `M`
    /// puntos (y legítimo, si se pide) nace como gaussiana y sus puntos toman
    /// su etiqueta.
    fn siembra_grupos(&mut self, idx: &[usize], c: usize, exige_legitimidad: bool) {
        let puntos: Vec<&[f64]> = idx.iter().map(|&i| self.ventana[i].x.as_slice()).collect();
        let grupos = pcm_agrupa(&puntos, c, self.p.umbral_pertenencia, self.p.factor_fusion, &mut self.rng);
        let dispersion_max = self
            .gaussianas
            .iter()
            .map(|g| g.dispersion)
            .fold(0.0_f64, f64::max);
        let mut nuevas = Vec::new();
        for miembros in grupos {
            if miembros.len() <= self.p.min_puntos {
                continue;
            }
            let pts: Vec<&[f64]> = miembros.iter().map(|&m| puntos[m]).collect();
            let Some(g) = Gaussiana::de_puntos(0, &pts) else { continue };
            if exige_legitimidad
                && dispersion_max > 0.0
                && g.dispersion > self.p.factor_dispersion * dispersion_max
            {
                continue;
            }
            //Un grupo cuyo centro cae dentro de una gaussiana es su borde, no
            //una estructura nueva.
            if exige_legitimidad
                && self.gaussianas.iter().any(|e| e.mahalanobis(&g.media) < self.p.umbral_distancia)
            {
                continue;
            }
            nuevas.push((g, miembros));
        }
        for (mut g, miembros) in nuevas {
            g.id = self.nuevo_id();
            for m in miembros {
                self.ventana[idx[m]].etiqueta = g.id;
            }
            self.gaussianas.push(g);
        }
    }

    pub fn inicializado(&self) -> bool {
        self.inicializado
    }

    /// Gaussianas creadas desde el arranque, con puntos en la ventana o no.
    pub fn n_grupos(&self) -> usize {
        self.gaussianas.len()
    }

    /// Gaussianas con algún punto dentro de la ventana actual.
    pub fn n_activos(&self) -> usize {
        self.gaussianas.iter().filter(|g| g.n > 0).count()
    }

    /// Tamaño del historial de anomalías: puntos de la ventana sin grupo.
    pub fn n_anomalias(&self) -> usize {
        self.ventana.iter().filter(|e| e.etiqueta == RUIDO).count()
    }

    pub fn n_ventana(&self) -> usize {
        self.ventana.len()
    }

    /// Los puntos de la ventana con su etiqueta actual, del más viejo al más nuevo.
    pub fn ventana(&self) -> impl Iterator<Item = (&[f64], i64)> {
        self.ventana.iter().map(|e| (e.x.as_slice(), e.etiqueta))
    }

    /// `(id, media, puntos en la ventana)` de cada gaussiana.
    pub fn grupos(&self) -> Vec<(i64, Vec<f64>, usize)> {
        self.gaussianas.iter().map(|g| (g.id, g.media.clone(), g.n)).collect()
    }
}

// ---------------------------------------------------------------------------
// PCM con fusión de prototipos coincidentes (el papel de AMPCM)
// ---------------------------------------------------------------------------

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Agrupa `puntos` con PCM (`C = c`, `m = 2`) y fusiona los prototipos
/// coincidentes. Devuelve los índices de cada grupo; los puntos con
/// pertenencia menor que `tn` a todos los grupos no aparecen (son anomalías).
fn pcm_agrupa(puntos: &[&[f64]], c: usize, tn: f64, fusion: f64, rng: &mut Rng) -> Vec<Vec<usize>> {
    let n = puntos.len();
    let c = c.min(n);
    if c == 0 {
        return Vec::new();
    }
    const ITER: usize = 100;
    const TOL: f64 = 1e-9;
    const EPS: f64 = 1e-12;

    //Sembrado k-means++ y FCM (m = 2) para tener prototipos y η razonables.
    let mut v = kmeanspp(puntos, c, rng);
    let mut u = vec![0.0; c * n];
    for _ in 0..ITER {
        for j in 0..n {
            let d2: Vec<f64> = (0..c).map(|i| dist2(puntos[j], &v[i]).max(EPS)).collect();
            let inv: f64 = d2.iter().map(|d| 1.0 / d).sum();
            for i in 0..c {
                u[i * n + j] = (1.0 / d2[i]) / inv;
            }
        }
        let mov = actualiza_prototipos(puntos, &u, &mut v);
        if mov < TOL {
            break;
        }
    }
    let eta: Vec<f64> = (0..c)
        .map(|i| {
            let (mut num, mut den) = (0.0, 0.0);
            for j in 0..n {
                let w = u[i * n + j] * u[i * n + j];
                num += w * dist2(puntos[j], &v[i]);
                den += w;
            }
            if den > 0.0 { (num / den).max(EPS) } else { EPS }
        })
        .collect();

    //PCM: uᵢⱼ = 1 / (1 + d²ᵢⱼ / ηᵢ), con η fijo.
    let pertenencias = |v: &[Vec<f64>], u: &mut [f64]| {
        for i in 0..c {
            for j in 0..n {
                u[i * n + j] = 1.0 / (1.0 + dist2(puntos[j], &v[i]) / eta[i]);
            }
        }
    };
    for _ in 0..ITER {
        pertenencias(&v, &mut u);
        let mov = actualiza_prototipos(puntos, &u, &mut v);
        if mov < TOL {
            break;
        }
    }
    pertenencias(&v, &mut u);

    //Fusión: dos prototipos con pertenencia ≥ ½ cada uno en el otro son el mismo grupo.
    let mut padre: Vec<usize> = (0..c).collect();
    fn raiz(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for a in 0..c {
        for b in a + 1..c {
            if dist2(&v[a], &v[b]) < fusion * eta[a].min(eta[b]) {
                let (ra, rb) = (raiz(&mut padre, a), raiz(&mut padre, b));
                padre[ra] = rb;
            }
        }
    }
    let raices: Vec<usize> = (0..c).map(|i| raiz(&mut padre, i)).collect();
    let mut unicas: Vec<usize> = raices.clone();
    unicas.sort_unstable();
    unicas.dedup();

    let mut grupos = vec![Vec::new(); unicas.len()];
    for j in 0..n {
        let (mejor, umax) = (0..c)
            .map(|i| (i, u[i * n + j]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("c > 0");
        if umax < tn {
            continue;
        }
        let g = unicas.binary_search(&raices[mejor]).expect("raíz conocida");
        grupos[g].push(j);
    }
    grupos
}

/// vᵢ = Σ u²ᵢⱼ xⱼ / Σ u²ᵢⱼ. Devuelve el mayor desplazamiento² de un prototipo.
fn actualiza_prototipos(puntos: &[&[f64]], u: &[f64], v: &mut [Vec<f64>]) -> f64 {
    let n = puntos.len();
    let d = puntos[0].len();
    let mut mov: f64 = 0.0;
    for (i, vi) in v.iter_mut().enumerate() {
        let mut num = vec![0.0; d];
        let mut den = 0.0;
        for j in 0..n {
            let w = u[i * n + j] * u[i * n + j];
            den += w;
            for k in 0..d {
                num[k] += w * puntos[j][k];
            }
        }
        if den > 0.0 {
            let nuevo: Vec<f64> = num.iter().map(|s| s / den).collect();
            mov = mov.max(dist2(&nuevo, vi));
            *vi = nuevo;
        }
    }
    mov
}

fn kmeanspp(puntos: &[&[f64]], c: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let n = puntos.len();
    let mut v = vec![puntos[rng.indice(n)].to_vec()];
    let mut d2: Vec<f64> = puntos.iter().map(|p| dist2(p, &v[0])).collect();
    while v.len() < c {
        let total: f64 = d2.iter().sum();
        let elegido = if total > 0.0 {
            let mut r = rng.uniforme() * total;
            let mut k = n - 1;
            for (j, &d) in d2.iter().enumerate() {
                if r < d {
                    k = j;
                    break;
                }
                r -= d;
            }
            k
        } else {
            rng.indice(n)
        };
        v.push(puntos[elegido].to_vec());
        for j in 0..n {
            d2[j] = d2[j].min(dist2(puntos[j], &v[v.len() - 1]));
        }
    }
    v
}

/// xorshift64*: basta para sembrar PCM de forma repetible, sin dependencias.
struct Rng(u64);

impl Rng {
    fn new(semilla: u64) -> Self {
        Rng(semilla.max(1))
    }
    fn siguiente(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniforme(&mut self) -> f64 {
        (self.siguiente() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn indice(&mut self, n: usize) -> usize {
        (self.uniforme() * n as f64) as usize % n
    }
}

#[cfg(test)]
mod pruebas {
    use super::*;

    /// Parámetros fijos de las pruebas del algoritmo: no dependen de la
    /// calibración (que cambia los valores por defecto), sino de lo que se
    /// prueba. Ventana de 1000 para que los datos sintéticos la llenen.
    fn prueba() -> Parametros {
        Parametros {
            ventana: 1000,
            min_puntos: 50,
            umbral_distancia: 2.5,
            umbral_pertenencia: 0.5,
            ..Default::default()
        }
    }

    fn normal(rng: &mut Rng) -> f64 {
        let u1 = rng.uniforme().max(1e-300);
        let u2 = rng.uniforme();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn nube(rng: &mut Rng, centro: (f64, f64), sigma: f64, n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|_| vec![centro.0 + sigma * normal(rng), centro.1 + sigma * normal(rng)])
            .collect()
    }

    /// Intercala varias nubes como llegarían espigas de neuronas distintas.
    fn mezcla(rng: &mut Rng, nubes: &[Vec<Vec<f64>>]) -> Vec<(Vec<f64>, usize)> {
        let mut todas: Vec<(Vec<f64>, usize)> = nubes
            .iter()
            .enumerate()
            .flat_map(|(k, c)| c.iter().map(move |p| (p.clone(), k)))
            .collect();
        for i in (1..todas.len()).rev() {
            let j = rng.indice(i + 1);
            todas.swap(i, j);
        }
        todas
    }

    /// Cada clase verdadera debe ir a una sola etiqueta mayoritaria, distinta
    /// para cada clase. Devuelve la fracción de puntos bien etiquetados.
    fn pureza(etiquetas: &[i64], verdad: &[usize], k: usize) -> (f64, Vec<i64>) {
        let mut mayoritaria = Vec::new();
        let mut bien = 0;
        for c in 0..k {
            let mut cuenta = std::collections::HashMap::new();
            for (e, v) in etiquetas.iter().zip(verdad) {
                if *v == c && *e >= 0 {
                    *cuenta.entry(*e).or_insert(0) += 1;
                }
            }
            let (e, n) = cuenta.into_iter().max_by_key(|x| x.1).unwrap_or((RUIDO, 0));
            mayoritaria.push(e);
            bien += n;
        }
        (bien as f64 / etiquetas.len() as f64, mayoritaria)
    }

    #[test]
    fn correccion_de_truncado() {
        //En 2D, P(χ²₂ < x) = 1 − e^{−x/2} y P(χ²₄ < x) = 1 − e^{−x/2}(1 + x/2).
        for td in [2.0_f64, 2.5, 3.0] {
            let x = td * td;
            let esperado = (1.0 - (-x / 2.0).exp()) / (1.0 - (-x / 2.0).exp() * (1.0 + x / 2.0));
            assert!((correccion_truncado(2, td) - esperado).abs() < 1e-9);
        }
        //Con umbral enorme no hay truncado.
        assert!((correccion_truncado(3, 50.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn una_sola_nube_estable_no_se_encoge_ni_se_parte() {
        let mut rng = Rng::new(23);
        let mut e = Erolsc::new(prueba()).unwrap();
        for _ in 0..50 {
            e.actualiza_lote(&nube(&mut rng, (0.5, 0.5), 0.03, 200)).unwrap();
        }
        assert_eq!(e.n_grupos(), 1, "grupos: {:?}", e.grupos());
        let g = &e.gaussianas[0];
        assert!((g.dispersion / 0.03 - 1.0).abs() < 0.15, "dispersión {} frente a 0.03", g.dispersion);
    }

    #[test]
    fn la_ventana_por_defecto_es_4000() {
        assert_eq!(Parametros::default().ventana, 4000);
        assert!(Parametros::default().valida().is_ok());
    }

    #[test]
    fn inicializa_con_la_ventana_y_encuentra_los_grupos() {
        let mut rng = Rng::new(7);
        let nubes = vec![
            nube(&mut rng, (0.2, 0.2), 0.02, 800),
            nube(&mut rng, (0.7, 0.3), 0.03, 600),
            nube(&mut rng, (0.4, 0.8), 0.02, 600),
        ];
        let datos = mezcla(&mut rng, &nubes);
        let mut e = Erolsc::new(prueba()).unwrap();
        let mut etiquetas = Vec::new();
        for lote in datos.chunks(200) {
            let pts: Vec<Vec<f64>> = lote.iter().map(|p| p.0.clone()).collect();
            etiquetas.extend(e.actualiza_lote(&pts).unwrap());
        }
        assert!(e.inicializado());
        assert_eq!(e.n_grupos(), 3, "grupos: {:?}", e.grupos());
        //Los 4 primeros lotes salen antes de que la ventana se llene: sin clasificar.
        assert!(etiquetas[..800].iter().all(|&x| x == SIN_CLASIFICAR));
        let verdad: Vec<usize> = datos.iter().map(|p| p.1).collect();
        //Con Td = 2.5 la cola de cada gaussiana (≈ 4 %) queda como ruido a
        //propósito: se mide aparte la pureza de lo etiquetado y el ruido.
        let (et, ve) = (&etiquetas[800..], &verdad[800..]);
        let ruido = et.iter().filter(|&&x| x == RUIDO).count() as f64 / et.len() as f64;
        assert!(ruido < 0.10, "ruido {ruido}");
        let (p, may) = pureza(et, ve, 3);
        assert!(p / (1.0 - ruido) > 0.99, "pureza sin ruido {}", p / (1.0 - ruido));
        assert_ne!(may[0], may[1]);
        assert_ne!(may[1], may[2]);
    }

    #[test]
    fn antes_de_llenar_la_ventana_no_clasifica() {
        let mut rng = Rng::new(3);
        let pts = nube(&mut rng, (0.5, 0.5), 0.02, 999);
        let mut e = Erolsc::new(prueba()).unwrap();
        let et = e.actualiza_lote(&pts).unwrap();
        assert!(!e.inicializado());
        assert!(et.iter().all(|&x| x == SIN_CLASIFICAR));
    }

    #[test]
    fn detecta_varias_estructuras_emergentes_a_la_vez() {
        let mut rng = Rng::new(11);
        let mut e = Erolsc::new(prueba()).unwrap();
        //Arranque: un solo grupo.
        let inicial = nube(&mut rng, (0.2, 0.8), 0.02, 1000);
        e.actualiza_lote(&inicial).unwrap();
        assert_eq!(e.n_grupos(), 1);
        //Luego aparecen tres a la vez, junto con el primero.
        let nubes = vec![
            nube(&mut rng, (0.2, 0.8), 0.02, 300),
            nube(&mut rng, (0.2, 0.2), 0.02, 300),
            nube(&mut rng, (0.5, 0.25), 0.02, 300),
            nube(&mut rng, (0.8, 0.2), 0.02, 300),
        ];
        let datos = mezcla(&mut rng, &nubes);
        for lote in datos.chunks(200) {
            let pts: Vec<Vec<f64>> = lote.iter().map(|p| p.0.clone()).collect();
            e.actualiza_lote(&pts).unwrap();
        }
        assert_eq!(e.n_grupos(), 4, "grupos: {:?}", e.grupos());
        //Cada nube nueva tiene su propio grupo, distinto del inicial.
        let ids: Vec<i64> = [(0.2, 0.2), (0.5, 0.25), (0.8, 0.2)]
            .iter()
            .map(|c| {
                e.grupos()
                    .into_iter()
                    .min_by(|a, b| dist2(&a.1, &[c.0, c.1]).total_cmp(&dist2(&b.1, &[c.0, c.1])))
                    .unwrap()
                    .0
            })
            .collect();
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert!(ids.iter().all(|&i| i != 0));
    }

    #[test]
    fn las_anomalias_aisladas_quedan_como_ruido() {
        let mut rng = Rng::new(5);
        let mut e = Erolsc::new(prueba()).unwrap();
        e.actualiza_lote(&nube(&mut rng, (0.3, 0.3), 0.02, 1000)).unwrap();
        let mut lote = nube(&mut rng, (0.3, 0.3), 0.02, 190);
        let lejanos: Vec<Vec<f64>> = (0..10)
            .map(|k| vec![0.6 + 0.04 * k as f64, 0.9 - 0.08 * k as f64])
            .collect();
        lote.extend(lejanos);
        let et = e.actualiza_lote(&lote).unwrap();
        assert!(et[190..].iter().all(|&x| x == RUIDO), "{:?}", &et[190..]);
        assert_eq!(e.n_grupos(), 1);
    }

    #[test]
    fn la_ventana_sigue_a_un_grupo_que_deriva() {
        let mut rng = Rng::new(13);
        let mut e = Erolsc::new(prueba()).unwrap();
        //Una neurona que se mueve de x = 0.2 a x = 0.6, poco a poco.
        let pasos = 40;
        for k in 0..pasos {
            let x = 0.2 + 0.4 * k as f64 / (pasos - 1) as f64;
            e.actualiza_lote(&nube(&mut rng, (x, 0.5), 0.02, 200)).unwrap();
        }
        assert_eq!(e.n_ventana(), 1000);
        let (id, media, n) = e.grupos()[0].clone();
        assert_eq!(id, 0, "la deriva no debe partir el grupo: {:?}", e.grupos());
        //La media es la de la ventana (últimos 5 lotes), no la de toda la historia.
        assert!(media[0] > 0.55, "media {media:?}");
        assert!(n > 950);
    }

    #[test]
    fn misma_semilla_mismo_resultado() {
        let mut rng = Rng::new(17);
        let nubes = vec![
            nube(&mut rng, (0.2, 0.2), 0.03, 700),
            nube(&mut rng, (0.7, 0.7), 0.03, 700),
        ];
        let pts: Vec<Vec<f64>> = mezcla(&mut rng, &nubes).into_iter().map(|p| p.0).collect();
        let corre = || {
            let mut e = Erolsc::new(prueba()).unwrap();
            pts.chunks(200).flat_map(|l| e.actualiza_lote(l).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(corre(), corre());
    }

    #[test]
    fn rechaza_parametros_y_dimensiones_invalidas() {
        assert!(Erolsc::new(Parametros { ventana: 0, ..Default::default() }).is_err());
        assert!(Erolsc::new(Parametros { min_puntos: 1000, ..prueba() }).is_err());
        let mut e = Erolsc::new(prueba()).unwrap();
        e.actualiza_lote(&[vec![0.0, 0.0]]).unwrap();
        assert!(e.actualiza_lote(&[vec![0.0, 0.0, 0.0]]).is_err());
    }
}
