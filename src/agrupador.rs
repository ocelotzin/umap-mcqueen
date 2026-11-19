//! Clasificación en línea por **propagación de etiquetas por vecindad**.
//!
//! Una unidad no es un modelo (gaussiana, centroide, micro-cluster): son **sus
//! propias espigas recientes**. Se guardan las últimas `ventana` espigas con
//! etiqueta, y cada espiga nueva:
//!
//! 1. busca sus `vecinos` más cercanas en la ventana;
//! 2. si la distancia media a ellas supera el `radio` típico, está lejos de
//!    toda unidad: es ruido y va al búfer de nacimientos;
//! 3. si no, toma la etiqueta mayoritaria. Si el voto es claro (≥ `voto_minimo`)
//!    **entra a la ventana con esa etiqueta**; si es dudoso, se emite pero no
//!    entra, para que un borde ambiguo no arrastre etiquetas de una unidad a otra;
//! 4. la espiga más vieja sale de la ventana.
//!
//! El paso 3 es lo que da **continuidad**: cuando una neurona deriva, sus
//! espigas nuevas siguen teniendo de vecinas a sus espigas recientes, heredan la
//! etiqueta y la llevan a la posición nueva. Ninguna etiqueta se reescribe en
//! bloque. El paso 4 es el olvido: la ventana suelta las posiciones viejas sola,
//! sin parámetro de decaimiento.
//!
//! **Nacimientos.** HDBSCAN corre sólo sobre las espigas que quedaron como ruido.
//! Un grupo denso, de al menos `min_cluster_size`, y lejos de toda unidad (a más
//! de `factor_nacer` × `radio` de las espigas con etiqueta) nace como unidad
//! nueva: sus espigas entran a la ventana con un id nuevo y desde ahí crece por
//! propagación. Un grupo cerca de una unidad es su borde, y se descarta.
//!
//! **Auditoría.** Cada `auditoria` espigas, en otro hilo, HDBSCAN corre sobre
//! una muestra de la ventana y se compara con las etiquetas: si un cluster
//! reúne buena parte de dos etiquetas, se **propone** unirlas; si una etiqueta
//! cae repartida en dos clusters, se propone separarla. Una propuesta aislada
//! **no reetiqueta**: reescribir en bloque en cada reajuste es lo que rompía la
//! continuidad. Sólo cuando la misma unión se repite en `unir_tras` auditorías
//! seguidas (o la misma separación en `separar_tras`) se aplica, una sola vez,
//! sobre las etiquetas de la ventana: en una unión la etiqueta con menos
//! espigas pasa a la otra; en una separación el cluster mayor conserva el id y
//! los demás reciben ids nuevos. Cada cambio aplicado queda en `uniones` o
//! `separaciones`; con `unir_tras`/`separar_tras` en 0 nunca se aplican. La
//! misma auditoría recalcula el `radio`.
//!
//! Medido en un encaje real de 94 208 espigas (prototipo): el cambio de
//! etiqueta de un lote al siguiente sobre una sonda fija nunca pasó de 1.5 %
//! (con reajustes periódicos de HDBSCAN llegaba al 95 %, y al 11 % en el 1 % de
//! los lotes), con ARI 0.97–0.98 frente a HDBSCAN de cada ventana.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::thread;

use hdbscan::{Hdbscan, HdbscanHyperParams, NnAlgorithm};

use crate::mezcla::{ajusta_modelo, Modelo, ParamsSeparacion};

/// Etiqueta de una espiga sin unidad. Coincide con `LABEL_TRASH` de tridesclous.
pub const ETIQUETA_RUIDO: i64 = -1;

/// Espigas con etiqueta que se recuerdan.
pub const VENTANA: usize = 10_000;
/// Vecinas que votan la etiqueta de cada espiga.
pub const VECINOS: usize = 10;
/// Fracción mínima de votos para que una espiga entre a la ventana con su
/// etiqueta (con 10 vecinas, 7). Por debajo se emite, pero no se propaga.
pub const VOTO_MINIMO: f64 = 0.7;
/// Espigas del arranque: el primer HDBSCAN las agrupa, y esas mismas espigas
/// reciben ya su etiqueta. Con 1 000, los 5 primeros lotes de 200 (16 de 64 en
/// `tdc_umap`) salían enteros como ruido; con 200, en un encaje real, ningún
/// lote pasa del 8 % de ruido y la partición final es la misma (sin la unión
/// que con 1 000 hacía falta para juntar las dos mitades de una neurona).
pub const ARRANQUE: usize = 200;
/// `min_cluster_size` del HDBSCAN del arranque: con 200 espigas, 40 dejaba sin
/// agrupar a las unidades medianas (lotes con 18–21 % de ruido); con 15, 1–8 %.
pub const MIN_CLUSTER_ARRANQUE: usize = 15;
/// Espigas mínimas para el primer agrupamiento y para que nazca una unidad.
pub const MIN_CLUSTER_SIZE: usize = 40;
/// Vecinos para la distancia de núcleo de HDBSCAN.
pub const MIN_SAMPLES: usize = 10;
/// Una espiga es ruido si su distancia media a las vecinas supera este
/// percentil de la misma medida entre espigas de la ventana.
pub const PERCENTIL_RADIO: f64 = 0.99;
/// Un grupo de espigas sin unidad nace como unidad sólo si su distancia mediana
/// a la espiga con etiqueta más cercana supera `factor_nacer` × `radio`. Por
/// debajo es el borde de una unidad que ya existe (con 1.0 los bordes nacían
/// como unidades: el prototipo llegó a 8 etiquetas con 3 neuronas).
pub const FACTOR_NACER: f64 = 2.0;
/// Espigas sin unidad que se guardan para buscar nacimientos.
pub const BUFER: usize = 1_000;
/// Espigas sin unidad mínimas para intentar un nacimiento.
pub const BUFER_MINIMO: usize = 300;
/// Cada cuántas espigas se audita (en otro hilo).
pub const AUDITORIA: usize = 5_000;
/// Espigas de la ventana que mira la auditoría.
pub const MUESTRA_AUDITORIA: usize = 2_500;
/// En la auditoría, fracción de una etiqueta que tiene que caer en un cluster
/// para contar como «está ahí» (y proponer unirla o separarla).
pub const FRACCION_AUDITORIA: f64 = 0.3;
/// Para unir dos etiquetas sin preguntar, cada una tiene que tener al menos
/// esta fracción dentro del mismo cluster, y en `UNIR_TRAS` auditorías
/// seguidas. Hace falta porque la propagación nunca une por sí sola: en un
/// encaje real el arranque partió una neurona en dos etiquetas (89 % y 11 %), y
/// la auditoría lo propuso en cada una de las 18 auditorías de la sesión.
pub const FRACCION_UNIR: f64 = 0.6;
/// Auditorías seguidas con la misma unión antes de aplicarla. Una cercanía
/// pasajera (dos neuronas que se acercan por deriva) no basta.
pub const UNIR_TRAS: u32 = 3;
/// Para separar una etiqueta sin preguntar, tiene que ser **dueña** de al menos
/// dos clusters de la auditoría: en cada uno, la mitad o más de sus espigas son
/// de ella, y tiene ahí al menos esta fracción de las suyas (y no menos de
/// `min_cluster_size` escalado). Es el espejo de las uniones: si una sesión
/// empieza a mitad de la grabación, el arranque puede no reunir espigas de una
/// unidad pequeña, la propagación las pega a la vecina y esa unidad nunca nace
/// (en un encaje real, una del 8.8 % quedó dentro de la del 60 %; la regla
/// laxa de «≥30 % en cada cluster» ni siquiera lo proponía, porque era el 13 %).
pub const FRACCION_SEPARAR: f64 = 0.05;
/// Auditorías seguidas con la misma separación antes de aplicarla.
pub const SEPARAR_TRAS: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct ParamsAgrupador {
    pub ventana: usize,
    pub vecinos: usize,
    pub voto_minimo: f64,
    pub arranque: usize,
    pub min_cluster_arranque: usize,
    pub min_cluster_size: usize,
    pub min_samples: usize,
    pub percentil_radio: f64,
    pub factor_nacer: f64,
    pub bufer: usize,
    pub auditoria: usize,
    /// La auditoría en otro hilo (por defecto). En síncrono es determinista.
    pub auditoria_asincrona: bool,
    /// Auditorías seguidas que tiene que repetirse una unión clara para
    /// aplicarse sola. 0 = nunca: sólo se propone.
    pub unir_tras: u32,
    /// Lo mismo para las separaciones claras. 0 = nunca: sólo se propone.
    pub separar_tras: u32,
    /// Separación local con mezcla de gaussianas sobre las formas de onda de
    /// cada unidad (ver `mezcla`). Sólo actúa si las espigas llegan con su forma
    /// de onda (`etiqueta_con_formas`); si no, o con `None`, la auditoría separa
    /// con HDBSCAN sobre el encaje como antes.
    pub separacion: Option<ParamsSeparacion>,
}

impl Default for ParamsAgrupador {
    fn default() -> Self {
        Self {
            ventana: VENTANA,
            vecinos: VECINOS,
            voto_minimo: VOTO_MINIMO,
            arranque: ARRANQUE,
            min_cluster_arranque: MIN_CLUSTER_ARRANQUE,
            min_cluster_size: MIN_CLUSTER_SIZE,
            min_samples: MIN_SAMPLES,
            percentil_radio: PERCENTIL_RADIO,
            factor_nacer: FACTOR_NACER,
            bufer: BUFER,
            auditoria: AUDITORIA,
            auditoria_asincrona: true,
            unir_tras: UNIR_TRAS,
            separar_tras: SEPARAR_TRAS,
            separacion: Some(ParamsSeparacion::default()),
        }
    }
}

impl ParamsAgrupador {
    pub fn valida(&self) -> Result<(), String> {
        if self.vecinos < 1 || self.min_cluster_size < 2 || self.min_samples < 1 {
            return Err("vecinos >= 1, min_cluster_size >= 2 y min_samples >= 1".into());
        }
        if self.ventana < self.arranque || self.arranque < 2 * self.min_cluster_arranque || self.min_cluster_arranque < 2 {
            return Err(format!(
                "hace falta ventana ({}) >= arranque ({}) >= 2·min_cluster_arranque ({}) >= 4",
                self.ventana, self.arranque, self.min_cluster_arranque
            ));
        }
        if !(0.0..=1.0).contains(&self.voto_minimo) || !(0.5..1.0).contains(&self.percentil_radio) {
            return Err("voto_minimo en [0, 1] y percentil_radio en [0.5, 1)".into());
        }
        if self.factor_nacer <= 0.0 || self.bufer < BUFER_MINIMO || self.auditoria == 0 {
            return Err(format!("factor_nacer > 0, bufer >= {BUFER_MINIMO} y auditoria > 0"));
        }
        Ok(())
    }
}

/// Lo que propone la auditoría. Nunca se aplica sola.
#[derive(Debug, Clone, PartialEq)]
pub enum Propuesta {
    /// Un cluster de HDBSCAN reúne buena parte de estas dos etiquetas.
    Unir(i64, i64),
    /// Esta etiqueta cae repartida en dos clusters de HDBSCAN.
    Separar(i64),
}

/// Resultado de una auditoría (calculado sin estado, puede ir en otro hilo).
struct Auditoria {
    radio: Option<f64>,
    propuestas: Vec<Propuesta>,
    /// Uniones con evidencia fuerte (`FRACCION_UNIR`), candidatas a aplicarse.
    claras: Vec<(i64, i64)>,
    /// Separaciones claras: por etiqueta, sus espigas de la auditoría con el
    /// subgrupo de cada una.
    separaciones: Vec<(i64, Vec<Miembro>)>,
    /// Segundo nivel: por grupo, la mezcla ajustada a sus formas de onda y el
    /// subgrupo de cada espiga (por instante de llegada), o `None` si la mezcla
    /// ve una sola neurona.
    modelos: Vec<(i64, Option<(Modelo, Vec<(u64, usize)>)>)>,
}

/// Segundo nivel de un grupo: la mezcla que reparte sus espigas entre
/// subunidades, y el id que se emite para cada subgrupo.
struct ModeloGrupo {
    modelo: Modelo,
    ids: Vec<i64>,
}

/// Una espiga que la auditoría asignó a un subgrupo al proponer una separación.
struct Miembro {
    x: Vec<f32>,
    forma: Option<Vec<f32>>,
    t: u64,
    c: i32,
}

/// Una espiga guardada: su punto en el encaje, su forma de onda (si llegó con
/// ella), su instante de llegada (orden en el flujo) y la unidad que se emitió
/// para ella (su grupo, o la subunidad si el grupo tiene segundo nivel).
#[derive(Clone)]
struct Espiga {
    x: Vec<f32>,
    forma: Option<Vec<f32>>,
    t: u64,
    unidad: i64,
}

/// Lo que la auditoría necesita de la ventana para separar con la mezcla:
/// (etiqueta, espiga) de todas las espigas con forma de onda, y el tramo de
/// tiempo que cubre la ventana.
struct Locales {
    espigas: Vec<(i64, Espiga)>,
    tramo: (u64, u64),
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Las `k` menores distancias (al cuadrado) de `x` a `puntos`, con su índice,
/// ordenadas. Fuerza bruta: con 10 000 puntos en 2D, ~10 µs por espiga, exacto
/// y sin índice que mantener al entrar y salir espigas.
fn k_vecinos<'a>(x: &[f32], puntos: impl Iterator<Item = &'a [f32]>, k: usize, excluye: Option<usize>) -> Vec<(f32, usize)> {
    let mut mejores: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, p) in puntos.enumerate() {
        if Some(i) == excluye {
            continue;
        }
        let d = dist2(x, p);
        if mejores.len() < k || d < mejores[mejores.len() - 1].0 {
            let pos = mejores.partition_point(|m| m.0 <= d);
            mejores.insert(pos, (d, i));
            if mejores.len() > k {
                mejores.pop();
            }
        }
    }
    mejores
}

fn media_raiz(v: &[(f32, usize)]) -> f64 {
    if v.is_empty() { f64::INFINITY } else { v.iter().map(|m| (m.0 as f64).sqrt()).sum::<f64>() / v.len() as f64 }
}

/// Radio típico: percentil `p` de la distancia media a las `k` vecinas, medida
/// en una muestra de la ventana contra la ventana entera.
fn calcula_radio(puntos: &[Vec<f32>], k: usize, p: f64) -> Option<f64> {
    if puntos.len() <= k {
        return None;
    }
    let paso = (puntos.len() / 1_000).max(1);
    let mut m: Vec<f64> = (0..puntos.len())
        .step_by(paso)
        .map(|i| media_raiz(&k_vecinos(&puntos[i], puntos.iter().map(|v| v.as_slice()), k, Some(i))))
        .collect();
    m.sort_by(|a, b| a.total_cmp(b));
    Some(m[((m.len() - 1) as f64 * p).round() as usize])
}

fn hdbscan(datos: &[Vec<f32>], mcs: usize, ms: usize) -> Result<Vec<i32>, String> {
    let hp = HdbscanHyperParams::builder()
        .min_cluster_size(mcs)
        .min_samples(ms.min(mcs))
        .nn_algorithm(NnAlgorithm::KdTree)
        .build();
    Hdbscan::new(datos, hp).cluster().map_err(|e| format!("HDBSCAN: {e:?}"))
}

/// Compara HDBSCAN sobre una muestra de la ventana con sus etiquetas.
fn audita(puntos: Vec<Vec<f32>>, tiempos: Vec<u64>, etiquetas: Vec<i64>, p: ParamsAgrupador, locales: Option<Locales>) -> Auditoria {
    let radio = calcula_radio(&puntos, p.vecinos, p.percentil_radio);
    let mcs = ((p.min_cluster_size * puntos.len()) as f64 / p.ventana as f64).round().max(5.0) as usize;
    let Ok(cl) = hdbscan(&puntos, mcs, p.min_samples) else {
        return Auditoria { radio, propuestas: Vec::new(), claras: Vec::new(), separaciones: Vec::new(), modelos: Vec::new() };
    };
    // total[e] y conjunta[(cluster, e)]
    let mut total: HashMap<i64, usize> = HashMap::new();
    let mut conjunta: HashMap<(i32, i64), usize> = HashMap::new();
    for (&c, &e) in cl.iter().zip(&etiquetas) {
        *total.entry(e).or_default() += 1;
        if c >= 0 {
            *conjunta.entry((c, e)).or_default() += 1;
        }
    }
    let fraccion_en = |c: i32, e: i64| -> f64 {
        let t = total.get(&e).copied().unwrap_or(0);
        if t < mcs { 0.0 } else { conjunta.get(&(c, e)).copied().unwrap_or(0) as f64 / t as f64 }
    };
    let esta = |c: i32, e: i64| fraccion_en(c, e) >= p.fraccion();
    let mut claras = Vec::new();
    let clusters: Vec<i32> = {
        let mut v: Vec<i32> = cl.iter().copied().filter(|&c| c >= 0).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut etiquetas_u: Vec<i64> = total.keys().copied().collect();
    etiquetas_u.sort_unstable();
    let mut propuestas = Vec::new();
    for &c in &clusters {
        let dentro: Vec<i64> = etiquetas_u.iter().copied().filter(|&e| esta(c, e)).collect();
        for i in 0..dentro.len() {
            for j in i + 1..dentro.len() {
                propuestas.push(Propuesta::Unir(dentro[i], dentro[j]));
                if fraccion_en(c, dentro[i]) >= FRACCION_UNIR && fraccion_en(c, dentro[j]) >= FRACCION_UNIR {
                    claras.push((dentro[i], dentro[j]));
                }
            }
        }
    }
    let mut tam_cluster: HashMap<i32, usize> = HashMap::new();
    for &c in cl.iter().filter(|&&c| c >= 0) {
        *tam_cluster.entry(c).or_default() += 1;
    }
    let mut separaciones = Vec::new();
    // Con formas de onda, las separaciones las decide la mezcla (abajo).
    let hdbscan_separa = locales.is_none() || p.separacion.is_none();
    for &e in &etiquetas_u {
        if e < 0 || !hdbscan_separa {
            continue;
        }
        if clusters.iter().filter(|&&c| esta(c, e)).count() >= 2 {
            propuestas.push(Propuesta::Separar(e));
        }
        // Dueña: la mitad o más del cluster es de `e`, y `e` tiene ahí una parte
        // no despreciable de las suyas.
        let te = total.get(&e).copied().unwrap_or(0);
        let duenas: Vec<i32> = clusters
            .iter()
            .copied()
            .filter(|&c| {
                let k = conjunta.get(&(c, e)).copied().unwrap_or(0);
                k >= mcs && k as f64 >= FRACCION_SEPARAR * te as f64 && 2 * k >= tam_cluster[&c]
            })
            .collect();
        if duenas.len() >= 2 {
            if !propuestas.contains(&Propuesta::Separar(e)) {
                propuestas.push(Propuesta::Separar(e));
            }
            let muestra: Vec<Miembro> = puntos
                .iter()
                .zip(&tiempos)
                .zip(&etiquetas)
                .zip(&cl)
                .filter(|(((_, _), et), c)| **et == e && duenas.contains(*c))
                .map(|(((x, &t), _), &c)| Miembro { x: x.clone(), forma: None, t, c })
                .collect();
            separaciones.push((e, muestra));
        }
    }
    let mut modelos = Vec::new();
    if let (Some(loc), Some(ps)) = (locales, p.separacion.as_ref()) {
        let mut por_etiqueta: HashMap<i64, Vec<Espiga>> = HashMap::new();
        for (e, esp) in loc.espigas {
            if e >= 0 {
                por_etiqueta.entry(e).or_default().push(esp);
            }
        }
        let mut claves: Vec<i64> = por_etiqueta.keys().copied().collect();
        claves.sort_unstable();
        for e in claves {
            let esp = &por_etiqueta[&e];
            let formas: Vec<&[f32]> = esp.iter().map(|s| s.forma.as_deref().unwrap_or(&[])).collect();
            let t: Vec<u64> = esp.iter().map(|s| s.t).collect();
            let res = ajusta_modelo(&formas, &t, loc.tramo, ps).map(|(m, sub)| (m, t.iter().copied().zip(sub).collect()));
            if res.is_some() {
                propuestas.push(Propuesta::Separar(e));
            }
            modelos.push((e, res));
        }
    }
    Auditoria { radio, propuestas, claras, separaciones, modelos }
}

impl ParamsAgrupador {
    fn fraccion(&self) -> f64 {
        FRACCION_AUDITORIA
    }
}

/// Una etiqueta por espiga, en línea.
pub struct Agrupador {
    p: ParamsAgrupador,
    /// Espigas con etiqueta (las que votan).
    ventana: VecDeque<(Espiga, i64)>,
    /// Espigas por etiqueta dentro de la ventana.
    conteo: HashMap<i64, usize>,
    /// Espigas sin unidad, para nacimientos.
    bufer: VecDeque<Espiga>,
    /// Arranque: espigas antes del primer agrupamiento.
    inicio: Vec<Espiga>,
    /// Espigas recibidas: el instante de llegada de la siguiente.
    llegadas: u64,
    /// Segundo nivel de los grupos que lo tienen.
    modelos: HashMap<i64, ModeloGrupo>,
    /// Cambios de segundo nivel pendientes: por grupo, el sentido del cambio
    /// (2 = partir o cambiar el número de subunidades, 1 = volver a una sola
    /// unidad) y en cuántas auditorías seguidas se propuso.
    persistencia_mod: HashMap<i64, (usize, u32)>,
    radio: Option<f64>,
    siguiente_id: i64,
    n_hdbscan: u64,
    desde_auditoria: usize,
    desde_nacimiento: usize,
    pendiente: Option<mpsc::Receiver<Auditoria>>,
    propuestas: Vec<Propuesta>,
    /// Auditorías seguidas en que cada unión clara se repitió.
    persistencia: HashMap<(i64, i64), u32>,
    /// Uniones aplicadas: (etiqueta absorbida, etiqueta que queda).
    uniones: Vec<(i64, i64)>,
    /// Auditorías seguidas en que cada separación clara se repitió.
    persistencia_sep: HashMap<i64, u32>,
    /// Separaciones aplicadas: (etiqueta original, etiquetas nuevas).
    separaciones: Vec<(i64, Vec<i64>)>,
    /// Ruido en las últimas espigas: (era ruido) en orden.
    recientes: VecDeque<bool>,
    dim: Option<usize>,
}

impl Default for Agrupador {
    fn default() -> Self {
        Self::new(ParamsAgrupador::default()).expect("parámetros por defecto válidos")
    }
}

impl Agrupador {
    pub fn new(p: ParamsAgrupador) -> Result<Self, String> {
        p.valida()?;
        Ok(Self {
            ventana: VecDeque::with_capacity(p.ventana + 1),
            conteo: HashMap::new(),
            bufer: VecDeque::with_capacity(p.bufer + 1),
            inicio: Vec::new(),
            llegadas: 0,
            modelos: HashMap::new(),
            persistencia_mod: HashMap::new(),
            radio: None,
            siguiente_id: 0,
            n_hdbscan: 0,
            desde_auditoria: 0,
            desde_nacimiento: 0,
            pendiente: None,
            propuestas: Vec::new(),
            persistencia: HashMap::new(),
            uniones: Vec::new(),
            persistencia_sep: HashMap::new(),
            separaciones: Vec::new(),
            recientes: VecDeque::new(),
            dim: None,
            p,
        })
    }

    pub fn params(&self) -> &ParamsAgrupador {
        &self.p
    }

    fn entra(&mut self, x: Espiga, e: i64) {
        self.ventana.push_back((x, e));
        *self.conteo.entry(e).or_default() += 1;
        if self.ventana.len() > self.p.ventana {
            let (_, viejo) = self.ventana.pop_front().unwrap();
            if let Some(c) = self.conteo.get_mut(&viejo) {
                *c -= 1;
                if *c == 0 {
                    self.conteo.remove(&viejo);
                }
            }
        }
    }

    /// Etiqueta de una espiga sin aprender de ella: (etiqueta, voto claro).
    fn vota(&self, x: &[f32]) -> (i64, bool) {
        let Some(radio) = self.radio else { return (ETIQUETA_RUIDO, false) };
        let k = self.p.vecinos.min(self.ventana.len());
        if k == 0 {
            return (ETIQUETA_RUIDO, false);
        }
        let vec = k_vecinos(x, self.ventana.iter().map(|(p, _)| p.x.as_slice()), k, None);
        if media_raiz(&vec) > radio {
            return (ETIQUETA_RUIDO, false);
        }
        let mut votos: HashMap<i64, usize> = HashMap::new();
        for &(_, i) in &vec {
            *votos.entry(self.ventana[i].1).or_default() += 1;
        }
        // Mayoría; a igualdad, la del vecino más cercano (orden de `vec`).
        let max = *votos.values().max().unwrap();
        let e = vec.iter().map(|&(_, i)| self.ventana[i].1).find(|e| votos[e] == max).unwrap();
        (e, max as f64 >= self.p.voto_minimo * k as f64)
    }

    /// Clasifica una espiga con el estado actual, sin aprender de ella (sólo el
    /// primer nivel: sin forma de onda no hay subunidad).
    pub fn clasifica_punto(&self, x: &[f32]) -> i64 {
        self.vota(x).0
    }

    /// Unidad que se emite para una espiga del grupo `g`: el grupo, o la
    /// subunidad que elige su mezcla si el grupo tiene segundo nivel.
    fn unidad_de(&self, g: i64, forma: Option<&[f32]>) -> i64 {
        match (self.modelos.get(&g), forma) {
            (Some(mg), Some(f)) => mg.ids[mg.modelo.clasifica(f)],
            _ => g,
        }
    }

    /// Una etiqueta por espiga, en el mismo orden; `ETIQUETA_RUIDO` si no tiene
    /// unidad. Las espigas con NaN/inf salen como ruido y no se aprenden.
    pub fn etiqueta(&mut self, puntos: &[Vec<f32>]) -> Result<Vec<i64>, String> {
        self.etiqueta_con_formas(puntos, None)
    }

    /// Como `etiqueta`, con la forma de onda de cada espiga (en el mismo orden).
    /// Con ellas, la auditoría puede separar unidades con la mezcla de
    /// gaussianas (`ParamsAgrupador::separacion`).
    pub fn etiqueta_con_formas(&mut self, puntos: &[Vec<f32>], formas: Option<&[Vec<f32>]>) -> Result<Vec<i64>, String> {
        if let Some(f) = formas {
            if f.len() != puntos.len() {
                return Err(format!("{} puntos y {} formas de onda", puntos.len(), f.len()));
            }
        }
        self.recoge_auditoria();
        let mut etiquetas = vec![ETIQUETA_RUIDO; puntos.len()];
        // Espigas de este lote que esperaban el arranque: si arranca en este
        // lote, se etiquetan con lo recién aprendido en vez de salir como ruido.
        let mut esperando: Vec<usize> = Vec::new();
        for (i, x) in puntos.iter().enumerate() {
            let t = self.llegadas;
            self.llegadas += 1;
            if !x.iter().all(|v| v.is_finite()) {
                continue;
            }
            let mut espiga = Espiga { x: x.clone(), forma: formas.map(|f| f[i].clone()), t, unidad: ETIQUETA_RUIDO };
            match self.dim {
                None => self.dim = Some(x.len()),
                Some(d) if d != x.len() => return Err(format!("dimensión {}, se esperaba {d}", x.len())),
                _ => {}
            }
            if self.radio.is_none() {
                self.inicio.push(espiga);
                esperando.push(i);
                if self.inicio.len() >= self.p.arranque {
                    self.arranca()?;
                    if self.radio.is_some() {
                        for &j in &esperando {
                            etiquetas[j] = self.vota(&puntos[j]).0;
                        }
                        esperando.clear();
                    }
                }
                self.anota(self.radio.is_none());
                continue;
            }
            let (e, claro) = self.vota(x);
            espiga.unidad = self.unidad_de(e, espiga.forma.as_deref());
            etiquetas[i] = espiga.unidad;
            if e == ETIQUETA_RUIDO {
                self.bufer.push_back(espiga);
                if self.bufer.len() > self.p.bufer {
                    self.bufer.pop_front();
                }
            } else if claro {
                self.entra(espiga, e);
            }
            self.anota(e == ETIQUETA_RUIDO);
            self.desde_auditoria += 1;
            self.desde_nacimiento += 1;
        }
        if self.radio.is_some() {
            if self.bufer.len() >= BUFER_MINIMO && self.desde_nacimiento >= BUFER_MINIMO / 2 {
                self.intenta_nacimientos()?;
            }
            if self.desde_auditoria >= self.p.auditoria && self.pendiente.is_none() {
                self.lanza_auditoria();
            }
        }
        Ok(etiquetas)
    }

    fn anota(&mut self, ruido: bool) {
        self.recientes.push_back(ruido);
        if self.recientes.len() > 2_000 {
            self.recientes.pop_front();
        }
    }

    /// Primer agrupamiento: HDBSCAN sobre las espigas del arranque.
    fn arranca(&mut self) -> Result<(), String> {
        let datos = std::mem::take(&mut self.inicio);
        let xs: Vec<Vec<f32>> = datos.iter().map(|s| s.x.clone()).collect();
        let cl = hdbscan(&xs, self.p.min_cluster_arranque, self.p.min_samples)?;
        self.n_hdbscan += 1;
        let mut id_de: HashMap<i32, i64> = HashMap::new();
        for (x, &c) in datos.iter().zip(&cl) {
            if c >= 0 {
                let n = self.siguiente_id;
                let id = *id_de.entry(c).or_insert_with(|| n);
                if id == n {
                    self.siguiente_id += 1;
                }
                self.entra(Espiga { unidad: id, ..x.clone() }, id);
            } else {
                self.bufer.push_back(x.clone());
            }
        }
        let ventana: Vec<Vec<f32>> = self.ventana.iter().map(|(p, _)| p.x.clone()).collect();
        // Sin ningún grupo, el radio no se puede medir: se reintenta con más espigas.
        self.radio = calcula_radio(&ventana, self.p.vecinos, self.p.percentil_radio);
        if self.radio.is_none() {
            self.inicio = datos;
            self.inicio.drain(..self.p.arranque / 2);
        }
        Ok(())
    }

    /// HDBSCAN sobre las espigas sin unidad: un grupo lejos de toda unidad nace.
    fn intenta_nacimientos(&mut self) -> Result<(), String> {
        self.desde_nacimiento = 0;
        let datos: Vec<Espiga> = self.bufer.iter().cloned().collect();
        let xs: Vec<Vec<f32>> = datos.iter().map(|s| s.x.clone()).collect();
        let cl = hdbscan(&xs, self.p.min_cluster_size, self.p.min_samples)?;
        self.n_hdbscan += 1;
        let radio = self.radio.unwrap_or(f64::INFINITY);
        let mut grupos: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &c) in cl.iter().enumerate() {
            if c >= 0 {
                grupos.entry(c).or_default().push(i);
            }
        }
        let mut quitar = vec![false; datos.len()];
        let mut claves: Vec<i32> = grupos.keys().copied().collect();
        claves.sort_unstable();
        for c in claves {
            let miembros = &grupos[&c];
            // Distancia mediana de sus espigas a la espiga con etiqueta más cercana.
            let mut d: Vec<f64> = miembros
                .iter()
                .map(|&i| media_raiz(&k_vecinos(&xs[i], self.ventana.iter().map(|(p, _)| p.x.as_slice()), 1, None)))
                .collect();
            d.sort_by(|a, b| a.total_cmp(b));
            let mediana = d[d.len() / 2];
            if mediana > self.p.factor_nacer * radio {
                let id = self.siguiente_id;
                self.siguiente_id += 1;
                for &i in miembros {
                    self.entra(Espiga { unidad: id, ..datos[i].clone() }, id);
                }
            }
            // Nazca o sea el borde de una unidad, sale del búfer: ya se decidió.
            for &i in miembros {
                quitar[i] = true;
            }
        }
        self.bufer = datos.into_iter().zip(quitar).filter(|(_, q)| !q).map(|(x, _)| x).collect();
        Ok(())
    }

    fn lanza_auditoria(&mut self) {
        self.desde_auditoria = 0;
        let n = self.ventana.len();
        let paso = n.div_ceil(MUESTRA_AUDITORIA).max(1);
        let puntos: Vec<Vec<f32>> = self.ventana.iter().step_by(paso).map(|(p, _)| p.x.clone()).collect();
        let tiempos: Vec<u64> = self.ventana.iter().step_by(paso).map(|(p, _)| p.t).collect();
        let etiquetas: Vec<i64> = self.ventana.iter().step_by(paso).map(|(_, e)| *e).collect();
        let locales = if self.p.separacion.is_some() && self.ventana.iter().any(|(s, _)| s.forma.is_some()) {
            let espigas: Vec<(i64, Espiga)> = self.ventana.iter().filter(|(s, _)| s.forma.is_some()).map(|(s, e)| (*e, s.clone())).collect();
            let t0 = espigas.iter().map(|(_, s)| s.t).min().unwrap_or(0);
            let t1 = espigas.iter().map(|(_, s)| s.t).max().unwrap_or(0);
            Some(Locales { espigas, tramo: (t0, t1) })
        } else {
            None
        };
        let p = self.p.clone();
        if self.p.auditoria_asincrona {
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                let _ = tx.send(audita(puntos, tiempos, etiquetas, p, locales));
            });
            self.pendiente = Some(rx);
        } else {
            let a = audita(puntos, tiempos, etiquetas, p, locales);
            self.aplica_auditoria(a);
        }
    }

    fn recoge_auditoria(&mut self) {
        if let Some(rx) = &self.pendiente {
            match rx.try_recv() {
                Ok(a) => {
                    self.pendiente = None;
                    self.aplica_auditoria(a);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => self.pendiente = None,
            }
        }
    }

    fn aplica_auditoria(&mut self, a: Auditoria) {
        self.n_hdbscan += 1;
        // El radio del arranque sale de pocas espigas; la primera auditoría lo
        // corrige con la ventana ya llena, y ese único ajuste mueve la frontera
        // del ruido (en la prueba de continuidad, 7.7 % de la sonda una vez).
        // Probé recalibrarlo mientras se llena la ventana y mezclarlo: el
        // salto bajaba a 5.1 %, pero el radio más estrecho hacía nacer
        // unidades espurias en un encaje real (5 ids en vez de 3).
        if a.radio.is_some() {
            self.radio = a.radio;
        }
        self.propuestas = a.propuestas;
        // Uniones claras: se cuentan las auditorías seguidas; a las
        // `unir_tras`, la etiqueta con menos espigas pasa a la otra. Es un
        // cambio de una vez y explícito (queda en `uniones`), no periódico.
        let anterior = std::mem::take(&mut self.persistencia);
        for par in a.claras {
            let n = anterior.get(&par).copied().unwrap_or(0) + 1;
            self.persistencia.insert(par, n);
        }
        let unidas_antes = self.uniones.len();
        let mut listas: Vec<(i64, i64)> = if self.p.unir_tras == 0 {
            Vec::new()
        } else {
            self.persistencia.iter().filter(|&(_, &n)| n >= self.p.unir_tras).map(|(&par, _)| par).collect()
        };
        listas.sort_unstable();
        for (a_, b_) in listas {
            let (ca, cb) = (self.conteo.get(&a_).copied().unwrap_or(0), self.conteo.get(&b_).copied().unwrap_or(0));
            if ca == 0 || cb == 0 {
                continue; // una ya se unió a otra en esta misma ronda
            }
            let (queda, absorbida) = if ca >= cb { (a_, b_) } else { (b_, a_) };
            // El segundo nivel de los dos se descarta: la mezcla del grupo unido
            // se vuelve a aprender en las auditorías siguientes.
            self.modelos.remove(&queda);
            self.modelos.remove(&absorbida);
            self.persistencia_mod.remove(&queda);
            self.persistencia_mod.remove(&absorbida);
            for (s, e) in self.ventana.iter_mut() {
                if *e == absorbida || *e == queda {
                    *e = queda;
                    s.unidad = queda;
                }
            }
            let n = self.conteo.remove(&absorbida).unwrap_or(0);
            *self.conteo.entry(queda).or_default() += n;
            self.uniones.push((absorbida, queda));
            self.persistencia.retain(|&(x, y), _| x != absorbida && y != absorbida);
        }

        // Separaciones claras, igual: a las `separar_tras` auditorías seguidas,
        // las espigas de la etiqueta que caen en el cluster menor reciben un id
        // nuevo (cada espiga de la ventana va con la espiga de la muestra más
        // cercana). Una etiqueta que se acaba de unir no se separa en la misma
        // ronda.
        let tocadas: Vec<i64> = self.uniones[unidas_antes..].iter().flat_map(|&(a, b)| [a, b]).collect();
        let anterior = std::mem::take(&mut self.persistencia_sep);
        for (e, _) in &a.separaciones {
            self.persistencia_sep.insert(*e, anterior.get(e).copied().unwrap_or(0) + 1);
        }
        if self.p.separar_tras == 0 {
            return;
        }
        let modelos_nuevos = a.modelos;
        for (e, muestra) in a.separaciones {
            if self.persistencia_sep.get(&e).copied().unwrap_or(0) < self.p.separar_tras
                || tocadas.contains(&e)
                || !self.conteo.contains_key(&e)
            {
                continue;
            }
            // El cluster con más espigas conserva el id; cada otro recibe uno nuevo.
            let mut tam: HashMap<i32, usize> = HashMap::new();
            for m in &muestra {
                *tam.entry(m.c).or_default() += 1;
            }
            // Cada espiga de la ventana toma el subgrupo con que la vio la
            // auditoría; las que llegaron después, el de la más cercana (en
            // forma de onda si hay, si no en el encaje).
            let por_t: HashMap<u64, i32> = muestra.iter().map(|m| (m.t, m.c)).collect();
            let mayor = *tam.iter().max_by_key(|&(c, &k)| (k, -c)).unwrap().0;
            let mut id_de: HashMap<i32, i64> = HashMap::new();
            let mut nuevas = Vec::new();
            let mut otros: Vec<i32> = tam.keys().copied().filter(|&c| c != mayor).collect();
            otros.sort_unstable();
            for c in otros {
                id_de.insert(c, self.siguiente_id);
                nuevas.push(self.siguiente_id);
                self.siguiente_id += 1;
            }
            for (s, et) in self.ventana.iter_mut() {
                if *et != e {
                    continue;
                }
                let c = por_t.get(&s.t).copied().unwrap_or_else(|| {
                    let d = |m: &Miembro| match (&s.forma, &m.forma) {
                        (Some(a), Some(b)) => dist2(a, b),
                        _ => dist2(&s.x, &m.x),
                    };
                    muestra.iter().min_by(|a, b| d(a).total_cmp(&d(b))).map(|m| m.c).unwrap_or(mayor)
                });
                if let Some(&nuevo) = id_de.get(&c) {
                    *et = nuevo;
                    s.unidad = nuevo;
                }
            }
            self.modelos.remove(&e);
            self.conteo.clear();
            for (_, et) in &self.ventana {
                *self.conteo.entry(*et).or_default() += 1;
            }
            self.persistencia_sep.remove(&e);
            self.separaciones.push((e, nuevas));
        }

        // Segundo nivel. Un cambio en el número de subunidades de un grupo
        // (crear su mezcla, quitarla o cambiar cuántas tiene) se aplica tras
        // `separar_tras` auditorías seguidas que lo propongan; con el mismo
        // número, la mezcla se reajusta en cada auditoría (así sigue la deriva)
        // y los ids pasan de la mezcla vieja a la nueva por solapamiento.
        for (g, res) in modelos_nuevos {
            if tocadas.contains(&g) || !self.conteo.contains_key(&g) {
                continue;
            }
            let n_viejo = self.modelos.get(&g).map_or(1, |m| m.ids.len());
            let n_nuevo = res.as_ref().map_or(1, |(m, _)| m.n_subgrupos());
            if n_nuevo != n_viejo {
                // Lo que persiste es el sentido del cambio (partir, o volver a
                // una sola unidad), no el número exacto: de una auditoría a
                // otra la mezcla puede ver 2 o 3 subgrupos en el mismo grupo, y
                // exigir el mismo número dejaba sin separar grupos que la mezcla
                // proponía partir en cada auditoría. Se instala la más reciente.
                let sentido = n_nuevo.min(2);
                let pend = self.persistencia_mod.entry(g).or_insert((sentido, 0));
                if pend.0 != sentido {
                    *pend = (sentido, 0);
                }
                pend.1 += 1;
                if pend.1 < self.p.separar_tras {
                    continue;
                }
            }
            self.persistencia_mod.remove(&g);
            match res {
                None => self.quita_segundo_nivel(g),
                Some((modelo, por_t)) => self.pone_segundo_nivel(g, modelo, por_t.into_iter().collect()),
            }
        }
    }

    /// El grupo `g` vuelve a ser una sola unidad, con el id del grupo.
    fn quita_segundo_nivel(&mut self, g: i64) {
        let Some(mg) = self.modelos.remove(&g) else { return };
        for &id in mg.ids.iter().filter(|&&i| i != g) {
            self.uniones.push((id, g));
        }
        for (s, e) in self.ventana.iter_mut() {
            if *e == g {
                s.unidad = g;
            }
        }
    }

    /// Instala (o reajusta) la mezcla del grupo `g`. Cada subgrupo nuevo hereda
    /// el id de la unidad vieja con que más espigas comparte en la ventana; los
    /// que no heredan ninguno reciben un id nuevo.
    fn pone_segundo_nivel(&mut self, g: i64, modelo: Modelo, por_t: HashMap<u64, usize>) {
        let k = modelo.n_subgrupos();
        let sub_de = |s: &Espiga| por_t.get(&s.t).copied().unwrap_or_else(|| s.forma.as_deref().map_or(0, |f| modelo.clasifica(f)));
        let mut solape: HashMap<(usize, i64), usize> = HashMap::new();
        for (s, e) in &self.ventana {
            if *e == g {
                *solape.entry((sub_de(s), s.unidad)).or_default() += 1;
            }
        }
        let mut pares: Vec<((usize, i64), usize)> = solape.into_iter().collect();
        pares.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut ids: Vec<Option<i64>> = vec![None; k];
        let mut usados: Vec<i64> = Vec::new();
        for ((sub, id), _) in pares {
            if ids[sub].is_none() && id >= 0 && !usados.contains(&id) {
                ids[sub] = Some(id);
                usados.push(id);
            }
        }
        let mut nuevas = Vec::new();
        let ids: Vec<i64> = ids
            .into_iter()
            .map(|id| {
                id.unwrap_or_else(|| {
                    let n = self.siguiente_id;
                    self.siguiente_id += 1;
                    nuevas.push(n);
                    n
                })
            })
            .collect();
        for (s, e) in self.ventana.iter_mut() {
            if *e == g {
                s.unidad = ids[sub_de(s)];
            }
        }
        if !nuevas.is_empty() {
            self.separaciones.push((g, nuevas));
        }
        self.modelos.insert(g, ModeloGrupo { modelo, ids });
    }

    /// Separaciones aplicadas: (etiqueta original, etiquetas nuevas).
    pub fn separaciones(&self) -> &[(i64, Vec<i64>)] {
        &self.separaciones
    }

    /// Uniones aplicadas en la sesión: (etiqueta absorbida, etiqueta que queda).
    pub fn uniones(&self) -> &[(i64, i64)] {
        &self.uniones
    }

    /// Espera la auditoría en curso (p. ej. al terminar un archivo).
    pub fn espera_ajuste(&mut self) -> Result<(), String> {
        if let Some(rx) = self.pendiente.take() {
            let a = rx.recv().map_err(|_| "el hilo de auditoría terminó sin resultado".to_string())?;
            self.aplica_auditoria(a);
        }
        Ok(())
    }

    /// Lo que propuso la última auditoría (unir o separar etiquetas). Nada se
    /// aplica solo.
    pub fn propuestas(&self) -> &[Propuesta] {
        &self.propuestas
    }

    /// Unidades (grupos, o sus subunidades si tienen segundo nivel) con al
    /// menos `min_cluster_size` espigas en la ventana.
    pub fn n_unidades(&self) -> usize {
        let mut c: HashMap<i64, usize> = HashMap::new();
        for (s, _) in &self.ventana {
            *c.entry(s.unidad).or_default() += 1;
        }
        c.values().filter(|&&n| n >= self.p.min_cluster_size).count()
    }

    /// Grupos con segundo nivel: (grupo, ids de sus subunidades).
    pub fn segundo_nivel(&self) -> Vec<(i64, Vec<i64>)> {
        let mut v: Vec<(i64, Vec<i64>)> = self.modelos.iter().map(|(&g, m)| (g, m.ids.clone())).collect();
        v.sort_unstable();
        v
    }

    /// Ids distintos entregados en toda la sesión.
    pub fn n_ids_emitidos(&self) -> i64 {
        self.siguiente_id
    }

    /// Veces que corrió HDBSCAN (arranque, nacimientos y auditorías).
    pub fn n_ajustes(&self) -> u64 {
        self.n_hdbscan
    }

    /// Espigas con etiqueta en la ventana.
    pub fn n_ventana(&self) -> usize {
        self.ventana.len()
    }

    /// Fracción de ruido en las últimas 2 000 espigas.
    pub fn fraccion_fuera(&self) -> f64 {
        if self.recientes.is_empty() {
            0.0
        } else {
            self.recientes.iter().filter(|&&r| r).count() as f64 / self.recientes.len() as f64
        }
    }

    /// Radio de ruido vigente.
    pub fn radio(&self) -> Option<f64> {
        self.radio
    }
}
