//! Los parámetros de DenStream, una sola vez para la CLI y para `tdc_umap`.
//!
//! **Las distancias de `clump` van al cuadrado.** `DenStream::new` usa
//! `SquaredEuclidean`, así que `EPSILON` y `MACRO_EPSILON` se comparan contra la
//! distancia euclídea **elevada al cuadrado** entre puntos del encaje normalizado
//! a [0,1]. El radio de un micro-cluster, en cambio, se compara sin elevar.
//!
//! Esto es lo que rompía la clasificación: con el `macro_epsilon` por defecto de
//! `clump` (2 × epsilon = 0.2) DBSCAN une centroides a √0.2 ≈ 0.45 de distancia,
//! casi media caja [0,1]², y encadena todas las unidades en un solo grupo.
//! Con `MACRO_EPSILON = 0.04` el alcance es √0.04 = 0.2. Ver
//! `examples/sonda_de_denstream.rs` y `tests/denstream.rs`.

use clump::{DenStream, NOISE};

/// Distancia² máxima para absorber un punto en un micro-cluster (√0.1 ≈ 0.32),
/// y radio máximo, sin elevar, del micro-cluster resultante (0.1).
pub const EPSILON: f32 = 0.11;
/// Vecinos que necesita un centroide para ser núcleo en el DBSCAN que agrupa
/// micro-clusters en unidades. Con 3, una neurona que ocupa uno o dos
/// micro-clusters nunca es núcleo.
pub const MIN_PTS: usize = 2;
pub const BETA: f64 = 0.5;
pub const LAMBDA: f64 = 0.0; // Un lambda mayor deshecha los clusters despues de un rato
pub const MU: f64 = 2.0;
/// Distancia² entre centroides para que DBSCAN los ponga en la misma unidad
/// (√0.04 = 0.2 en el encaje normalizado).
pub const MACRO_EPSILON: f32 = 0.04;

/// DenStream con los parámetros de arriba.
pub fn nuevo() -> DenStream {
    DenStream::new(EPSILON, MIN_PTS)
        .with_beta(BETA)
        .with_lambda(LAMBDA)
        .with_mu(MU)
        .with_macro_epsilon(MACRO_EPSILON)
}

/// Unidades que hay ahora: grupos de DBSCAN más micro-clusters aislados (que
/// DBSCAN deja como ruido pero `tdc_umap` trata como unidad propia).
pub fn n_unidades(ds: &DenStream) -> usize {
    let etiquetas = match ds.macro_cluster() {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let grupos = etiquetas
        .iter()
        .filter(|&&k| k != NOISE)
        .collect::<std::collections::HashSet<_>>()
        .len();
    grupos + etiquetas.iter().filter(|&&k| k == NOISE).count()
}

/// Distancia (sin elevar) entre centroides de unidad para que dos lotes
/// seguidos se consideren la misma unidad (ver `identidades`).
pub const TOLERANCIA: f32 = 0.15;
/// Lotes sin ver una unidad antes de olvidarla. Su id no se reutiliza.
pub const OLVIDO_LOTES: u64 = 200;

/// Etiqueta de una forma de onda sin unidad.
pub const ETIQUETA_RUIDO: i64 = -1;

/// DenStream + agrupamiento + ids estables: **una etiqueta por punto**.
///
/// Es lo que hacen igual la CLI y `tdc_umap`, para que las dos den las mismas
/// etiquetas con los mismos datos. Por lote:
///
/// 1. `update_batch` — DenStream aprende (absorbe, crea y poda micro-clusters).
/// 2. `predict_batch` — cada punto contra el estado FINAL. Los índices de
///    `update_batch` no valen: si la poda salta a mitad del lote, `retain`
///    recorre el `Vec` y los primeros apuntan a otro micro-cluster.
/// 3. `macro_cluster` — DBSCAN sobre los centroides agrupa micro-clusters en
///    unidades; un micro aislado puede ser unidad propia.
/// 4. `Identidades::asigna` — de grupo (posición, cambia con cada poda) a id
///    estable entre lotes.
pub struct Etiquetador {
    pub ds: DenStream,
    identidades: crate::identidades::Identidades,
    /// Un micro-cluster que DBSCAN deja como ruido se trata como unidad propia.
    /// Sin esto, una neurona compacta que cabe en un solo micro sale entera
    /// como ruido.
    pub micro_aislado_es_unidad: bool,
}

impl Default for Etiquetador {
    fn default() -> Self {
        Self::con(nuevo(), TOLERANCIA, OLVIDO_LOTES, true)
    }
}

impl Etiquetador {
    pub fn con(ds: DenStream, tolerancia: f32, olvido_lotes: u64, micro_aislado_es_unidad: bool) -> Self {
        Self {
            ds,
            identidades: crate::identidades::Identidades::new(tolerancia, olvido_lotes),
            micro_aislado_es_unidad,
        }
    }

    /// Una etiqueta por punto, en el mismo orden; `ETIQUETA_RUIDO` si no tiene
    /// unidad. Los puntos con NaN/inf salen como ruido sin tocar DenStream (que
    /// los rechaza con error y tiraría el lote entero).
    pub fn etiqueta(&mut self, puntos: &[Vec<f32>]) -> Result<Vec<i64>, String> {
        use std::collections::HashMap;

        let mut etiquetas = vec![ETIQUETA_RUIDO; puntos.len()];
        let validos: Vec<usize> = (0..puntos.len())
            .filter(|&i| puntos[i].iter().all(|x| x.is_finite()))
            .collect();
        if validos.is_empty() {
            return Ok(etiquetas);
        }
        let pv: Vec<Vec<f32>> = validos.iter().map(|&i| puntos[i].clone()).collect();

        // 1) Aprender.
        self.ds.update_batch(&pv).map_err(|e| e.to_string())?;

        // 2) Cada punto contra el estado final.
        let micro: Vec<usize> = if self.ds.n_clusters() == 0 {
            vec![NOISE; pv.len()]
        } else {
            self.ds.predict_batch(&pv).map_err(|e| e.to_string())?
        };

        // 3) Micro → grupo.
        let macro_de_micro = self.ds.macro_cluster().unwrap_or_default();
        let centroides = self.ds.centroids();
        let cuentas = self.ds.counts();
        let aislado = self.micro_aislado_es_unidad;
        // Clave: k >= 0 para un grupo de DBSCAN; -(j+2) para un micro aislado.
        let clave_de_micro = |j: usize| -> Option<i64> {
            match macro_de_micro.get(j) {
                Some(&k) if k != NOISE => Some(k as i64),
                Some(_) if aislado => Some(-(j as i64) - 2),
                _ => None,
            }
        };
        // Centroide de cada grupo, ponderado por los puntos de cada micro.
        let mut sumas: HashMap<i64, (Vec<f64>, f64)> = HashMap::new();
        for (j, c) in centroides.iter().enumerate() {
            if let Some(clave) = clave_de_micro(j) {
                let w = cuentas.get(j).copied().unwrap_or(1).max(1) as f64;
                let e = sumas.entry(clave).or_insert_with(|| (vec![0.0; c.len()], 0.0));
                for (s, &x) in e.0.iter_mut().zip(c) {
                    *s += w * x as f64;
                }
                e.1 += w;
            }
        }
        let mut claves: Vec<i64> = sumas.keys().copied().collect();
        claves.sort_unstable();
        let centroides_grupo: Vec<Vec<f32>> = claves
            .iter()
            .map(|k| {
                let (s, w) = &sumas[k];
                s.iter().map(|x| (x / w) as f32).collect()
            })
            .collect();

        // 4) Grupo → id estable.
        let ids = self.identidades.asigna(&centroides_grupo);
        let id_de_clave: HashMap<i64, i64> = claves.into_iter().zip(ids).collect();
        for (pos, &i) in validos.iter().enumerate() {
            if micro[pos] == NOISE {
                continue;
            }
            if let Some(id) = clave_de_micro(micro[pos]).and_then(|k| id_de_clave.get(&k)) {
                etiquetas[i] = *id;
            }
        }
        Ok(etiquetas)
    }

    /// Micro-clusters potenciales vivos.
    pub fn n_micro(&self) -> usize {
        self.ds.n_clusters()
    }

    /// Unidades vivas en la capa de identidades.
    pub fn n_unidades(&self) -> usize {
        self.identidades.n_unidades()
    }

    /// Ids distintos entregados en toda la sesión.
    pub fn n_ids_emitidos(&self) -> i64 {
        self.identidades.n_ids_emitidos()
    }
}
