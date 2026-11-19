//! Separación local de una unidad: **mezcla de gaussianas + BIC** en un PCA
//! local de sus formas de onda, con un **filtro temporal** contra la deriva.
//!
//! **Por qué no HDBSCAN.** Dos neuronas con formas de onda parecidas forman, en
//! un canal, dos nubes solapadas sin valle de densidad entre ellas: un método de
//! densidad las ve como un grupo. Un modelo de mezcla las separa por su forma y
//! su covarianza. Medido offline contra verdad de campo (híbridos de Janelia,
//! 16 canales, 66 neuronas): partir con HDBSCAN local o con un dip test casi no
//! parte nada; con GMM + BIC el ARI mediano sube de 0.45 a 0.50, pero parte de
//! más (89 unidades para 66 neuronas).
//!
//! **Filtro temporal.** Las separaciones falsas vienen sobre todo de la deriva:
//! una neurona que deriva forma una nube alargada que el BIC parte en «antes» y
//! «después». Dos neuronas distintas disparan a la vez; las dos mitades de una
//! que deriva ocupan tramos distintos. Si los perfiles temporales de dos
//! subgrupos se solapan menos que `coexistencia_minima`, se vuelven a juntar.
//! Con 0.5, el ARI mediano pasó de 0.45 a 0.63 (techo supervisado: 0.67) en 600 s
//! y de 0.45 a 0.60 (techo 0.71) en 1200 s. Un filtro por distancia entre
//! plantillas no sirvió: las neuronas que se funden difieren menos que el ruido.
//!
//! Todo es determinista (semilla fija) y sin dependencias: PCA por Jacobi, EM
//! con Cholesky en dimensión baja.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Parámetros de la separación local. Los valores por defecto son los
/// validados offline.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamsSeparacion {
    /// Componentes del PCA local.
    pub n_pca: usize,
    /// Máximo de componentes de la mezcla.
    pub k_max: usize,
    /// Un subgrupo cuenta si tiene al menos max(`min_abs`, `min_frac` × espigas).
    pub min_abs: usize,
    pub min_frac: f64,
    /// Dos subgrupos cuyos perfiles temporales se solapan menos que esto son la
    /// misma neurona que derivó y se juntan.
    pub coexistencia_minima: f64,
    /// Tramos del perfil temporal.
    pub tramos: usize,
}

impl Default for ParamsSeparacion {
    fn default() -> Self {
        Self { n_pca: 5, k_max: 4, min_abs: 100, min_frac: 0.05, coexistencia_minima: 0.5, tramos: 10 }
    }
}

impl ParamsSeparacion {
    fn min_tam(&self, n: usize) -> usize {
        self.min_abs.max((self.min_frac * n as f64) as usize)
    }
}

const SEMILLA: u64 = 0;
const N_INIT: usize = 2;
const REG_COVAR: f64 = 1e-6;
const MAX_ITER: usize = 100;
const TOL: f64 = 1e-3;
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Subgrupo de cada espiga de una unidad (0..m), o `None` si la unidad no se
/// parte. `tiempos` es el instante de llegada de cada espiga y `tramo` el
/// intervalo sobre el que se mide la coexistencia (lo que cubre la ventana).
pub fn separa_unidad(formas: &[&[f32]], tiempos: &[u64], tramo: (u64, u64), p: &ParamsSeparacion) -> Option<Vec<usize>> {
    ajusta_modelo(formas, tiempos, tramo, p).map(|(_, sub)| sub)
}

/// Lo que queda de una separación para clasificar espigas nuevas: la
/// proyección local y la mezcla, con el subgrupo de cada componente (varias
/// componentes pueden ser un mismo subgrupo si el filtro temporal las juntó).
#[derive(Clone)]
pub struct Modelo {
    proy: Proyeccion,
    mezcla: Mezcla,
    subgrupo_de: Vec<usize>,
    n_subgrupos: usize,
}

impl Modelo {
    /// Subgrupo (0..n_subgrupos) de una forma de onda: la componente más
    /// probable de la mezcla en el PCA local.
    pub fn clasifica(&self, forma: &[f32]) -> usize {
        if forma.len() != self.proy.media.len() {
            return 0;
        }
        let x = self.proy.aplica(forma);
        let mut lc = vec![0.0; self.mezcla.pesos.len()];
        self.mezcla.log_conjunta(&x, &mut lc);
        let c = (0..lc.len()).max_by(|&a, &b| lc[a].total_cmp(&lc[b])).unwrap_or(0);
        self.subgrupo_de[c]
    }

    pub fn n_subgrupos(&self) -> usize {
        self.n_subgrupos
    }
}

/// Como `separa_unidad`, y además el modelo con que clasificar espigas nuevas.
pub fn ajusta_modelo(formas: &[&[f32]], tiempos: &[u64], tramo: (u64, u64), p: &ParamsSeparacion) -> Option<(Modelo, Vec<usize>)> {
    let n = formas.len();
    let n_min = p.min_tam(n);
    if n < 2 * n_min || formas.is_empty() || formas[0].is_empty() {
        return None;
    }
    let (proy, x) = pca_local(formas, p.n_pca)?;
    let (mezcla, comp) = gmm_bic(&x, p.k_max, n_min)?;
    let sub = limpia(comp.clone(), &x, n_min)?;
    let sub = junta_por_tiempo(sub, tiempos, tramo, p);
    let n_subgrupos = sub.iter().max().map_or(0, |&s| s + 1);
    if n_subgrupos < 2 {
        return None;
    }
    // Subgrupo de cada componente: el de la mayoría de sus espigas.
    let k = mezcla.pesos.len();
    let mut votos = vec![vec![0usize; n_subgrupos]; k];
    for (&c, &s) in comp.iter().zip(&sub) {
        votos[c][s] += 1;
    }
    let subgrupo_de = votos.iter().map(|v| (0..n_subgrupos).max_by_key(|&s| v[s]).unwrap_or(0)).collect();
    Some((Modelo { proy, mezcla, subgrupo_de, n_subgrupos }, sub))
}

// ---------------------------------------------------------------- PCA local

/// Estandarización y ejes principales medidos sobre las espigas de una unidad.
#[derive(Clone)]
struct Proyeccion {
    media: Vec<f64>,
    sd: Vec<f64>,
    ejes: Vec<Vec<f64>>,
}

impl Proyeccion {
    fn aplica(&self, forma: &[f32]) -> Vec<f64> {
        let z: Vec<f64> = forma.iter().zip(&self.media).zip(&self.sd).map(|((&v, m), s)| (v as f64 - m) / s).collect();
        self.ejes.iter().map(|e| e.iter().zip(&z).map(|(a, b)| a * b).sum()).collect()
    }
}

/// Estandariza cada muestra (media 0, desviación 1 sobre estas espigas) y
/// proyecta sobre los `k` ejes principales.
fn pca_local(formas: &[&[f32]], k: usize) -> Option<(Proyeccion, Vec<Vec<f64>>)> {
    let n = formas.len();
    let d = formas[0].len();
    if formas.iter().any(|f| f.len() != d) {
        return None;
    }
    let mut media = vec![0.0; d];
    for f in formas {
        for (m, &v) in media.iter_mut().zip(f.iter()) {
            *m += v as f64;
        }
    }
    media.iter_mut().for_each(|m| *m /= n as f64);
    let mut sd = vec![0.0; d];
    for f in formas {
        for j in 0..d {
            sd[j] += (f[j] as f64 - media[j]).powi(2);
        }
    }
    sd.iter_mut().for_each(|s| *s = if *s > 1e-24 { (*s / n as f64).sqrt() } else { 1.0 });
    let z: Vec<Vec<f64>> = formas.iter().map(|f| (0..d).map(|j| (f[j] as f64 - media[j]) / sd[j]).collect()).collect();

    let mut cov = vec![vec![0.0; d]; d];
    for fila in &z {
        for a in 0..d {
            for b in a..d {
                cov[a][b] += fila[a] * fila[b];
            }
        }
    }
    for a in 0..d {
        for b in a..d {
            cov[a][b] /= (n - 1).max(1) as f64;
            cov[b][a] = cov[a][b];
        }
    }
    let (valores, vectores) = jacobi(cov);
    let mut orden: Vec<usize> = (0..d).collect();
    orden.sort_by(|&a, &b| valores[b].total_cmp(&valores[a]));
    let k = k.min(d).min(n.saturating_sub(1)).max(1);
    let ejes: Vec<Vec<f64>> = orden[..k].iter().map(|&c| (0..d).map(|r| vectores[r][c]).collect()).collect();
    let x = z.iter().map(|fila| ejes.iter().map(|e| e.iter().zip(fila).map(|(a, b)| a * b).sum()).collect()).collect();
    Some((Proyeccion { media, sd, ejes }, x))
}

/// Valores y vectores propios (en columnas) de una matriz simétrica, por
/// rotaciones de Jacobi cíclicas.
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let d = a.len();
    let mut v: Vec<Vec<f64>> = (0..d).map(|i| (0..d).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    let escala: f64 = a.iter().flatten().map(|x| x * x).sum::<f64>().max(1e-300);
    for _ in 0..100 {
        let fuera: f64 = (0..d).flat_map(|i| (0..d).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if fuera <= 1e-22 * escala {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for fila in v.iter_mut() {
                    let (vp, vq) = (fila[p], fila[q]);
                    fila[p] = c * vp - s * vq;
                    fila[q] = s * vp + c * vq;
                }
            }
        }
    }
    ((0..d).map(|i| a[i][i]).collect(), v)
}

// ---------------------------------------------------------------- GMM

#[derive(Clone)]
struct Mezcla {
    pesos: Vec<f64>,
    medias: Vec<Vec<f64>>,
    /// Factor de Cholesky (triangular inferior) de cada covarianza.
    chol: Vec<Vec<Vec<f64>>>,
}

fn cholesky(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let d = m.len();
    let mut l = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let v = m[i][i] - s;
                if v <= 0.0 || !v.is_finite() {
                    return None;
                }
                l[i][i] = v.sqrt();
            } else {
                l[i][j] = (m[i][j] - s) / l[j][j];
            }
        }
    }
    Some(l)
}

impl Mezcla {
    /// log(peso_c · N(x | c)) de cada componente.
    fn log_conjunta(&self, x: &[f64], out: &mut [f64]) {
        let d = x.len();
        let mut y = vec![0.0; d];
        for (c, o) in out.iter_mut().enumerate() {
            let l = &self.chol[c];
            let mu = &self.medias[c];
            let mut maha = 0.0;
            let mut logdet = 0.0;
            for i in 0..d {
                let s: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
                y[i] = (x[i] - mu[i] - s) / l[i][i];
                maha += y[i] * y[i];
                logdet += l[i][i].ln();
            }
            *o = self.pesos[c].ln() - 0.5 * (d as f64 * LN_2PI + maha) - logdet;
        }
    }

    /// Paso M a partir de responsabilidades (n × k).
    fn desde_resp(x: &[Vec<f64>], resp: &[Vec<f64>], k: usize) -> Option<Mezcla> {
        let n = x.len();
        let d = x[0].len();
        let mut pesos = vec![0.0; k];
        let mut medias = vec![vec![0.0; d]; k];
        for (xi, ri) in x.iter().zip(resp) {
            for c in 0..k {
                pesos[c] += ri[c];
                for j in 0..d {
                    medias[c][j] += ri[c] * xi[j];
                }
            }
        }
        for c in 0..k {
            pesos[c] += 10.0 * f64::EPSILON;
            medias[c].iter_mut().for_each(|m| *m /= pesos[c]);
        }
        let mut chol = Vec::with_capacity(k);
        for c in 0..k {
            let mut cov = vec![vec![0.0; d]; d];
            for (xi, ri) in x.iter().zip(resp) {
                if ri[c] < 1e-12 {
                    continue;
                }
                for a in 0..d {
                    let da = xi[a] - medias[c][a];
                    for b in 0..=a {
                        cov[a][b] += ri[c] * da * (xi[b] - medias[c][b]);
                    }
                }
            }
            for a in 0..d {
                for b in 0..=a {
                    cov[a][b] /= pesos[c];
                    cov[b][a] = cov[a][b];
                }
                cov[a][a] += REG_COVAR;
            }
            chol.push(cholesky(&cov)?);
        }
        pesos.iter_mut().for_each(|w| *w /= n as f64);
        Some(Mezcla { pesos, medias, chol })
    }
}

/// k-means++ y unas iteraciones de Lloyd, para iniciar la mezcla (como
/// scikit-learn, `init_params="kmeans"`).
fn kmeans(x: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = x.len();
    let d2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>();
    let mut centros = vec![x[rng.gen_range(0..n)].clone()];
    let mut dmin: Vec<f64> = x.iter().map(|xi| d2(xi, &centros[0])).collect();
    while centros.len() < k {
        let total: f64 = dmin.iter().sum();
        let mut r = rng.r#gen::<f64>() * total;
        let mut elegido = n - 1;
        for (i, &dm) in dmin.iter().enumerate() {
            r -= dm;
            if r <= 0.0 {
                elegido = i;
                break;
            }
        }
        centros.push(x[elegido].clone());
        let c = centros.last().unwrap().clone();
        for (dm, xi) in dmin.iter_mut().zip(x) {
            *dm = dm.min(d2(xi, &c));
        }
    }
    let mut asig = vec![0usize; n];
    for _ in 0..20 {
        let mut cambio = false;
        for (a, xi) in asig.iter_mut().zip(x) {
            let mejor = (0..k).min_by(|&p, &q| d2(xi, &centros[p]).total_cmp(&d2(xi, &centros[q]))).unwrap();
            cambio |= *a != mejor;
            *a = mejor;
        }
        let d = x[0].len();
        let mut suma = vec![vec![0.0; d]; k];
        let mut cuenta = vec![0usize; k];
        for (xi, &a) in x.iter().zip(&asig) {
            cuenta[a] += 1;
            suma[a].iter_mut().zip(xi).for_each(|(s, v)| *s += v);
        }
        for c in 0..k {
            if cuenta[c] > 0 {
                centros[c] = suma[c].iter().map(|s| s / cuenta[c] as f64).collect();
            }
        }
        if !cambio {
            break;
        }
    }
    asig
}

/// EM desde un k-means; devuelve la mezcla y su log-verosimilitud total.
fn ajusta(x: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Option<(Mezcla, f64)> {
    let n = x.len();
    let asig = kmeans(x, k, rng);
    let mut resp: Vec<Vec<f64>> = asig.iter().map(|&a| (0..k).map(|c| f64::from(u8::from(c == a))).collect()).collect();
    let mut m = Mezcla::desde_resp(x, &resp, k)?;
    let mut previa = f64::NEG_INFINITY;
    let mut lc = vec![0.0; k];
    let mut ll = 0.0;
    for _ in 0..MAX_ITER {
        ll = 0.0;
        for (xi, ri) in x.iter().zip(resp.iter_mut()) {
            m.log_conjunta(xi, &mut lc);
            let mx = lc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = mx + lc.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
            ll += lse;
            for c in 0..k {
                ri[c] = (lc[c] - lse).exp();
            }
        }
        if (ll / n as f64 - previa).abs() < TOL {
            break;
        }
        previa = ll / n as f64;
        m = Mezcla::desde_resp(x, &resp, k)?;
    }
    Some((m, ll))
}

/// Mezcla con 1..=`k_max` componentes (sin pasar de n / `n_min`); la de menor
/// BIC. Devuelve la componente más probable de cada espiga.
fn gmm_bic(x: &[Vec<f64>], k_max: usize, n_min: usize) -> Option<(Mezcla, Vec<usize>)> {
    let n = x.len();
    let d = x[0].len() as f64;
    let mut mejor: Option<(f64, Mezcla)> = None;
    for k in 1..=k_max {
        if n < k * n_min {
            break;
        }
        let mut rng = StdRng::seed_from_u64(SEMILLA + k as u64);
        let mut mejor_k: Option<(Mezcla, f64)> = None;
        for _ in 0..N_INIT {
            if let Some((m, ll)) = ajusta(x, k, &mut rng) {
                if mejor_k.as_ref().is_none_or(|(_, l)| ll > *l) {
                    mejor_k = Some((m, ll));
                }
            }
        }
        let Some((m, ll)) = mejor_k else { continue };
        let kf = k as f64;
        let libres = kf * d + kf * d * (d + 1.0) / 2.0 + kf - 1.0;
        let bic = -2.0 * ll + libres * (n as f64).ln();
        if mejor.as_ref().is_none_or(|(b, _)| bic < *b) {
            mejor = Some((bic, m));
        }
    }
    let (_, m) = mejor?;
    let k = m.pesos.len();
    let mut lc = vec![0.0; k];
    let comp = x
        .iter()
        .map(|xi| {
            m.log_conjunta(xi, &mut lc);
            (0..k).max_by(|&a, &b| lc[a].total_cmp(&lc[b])).unwrap()
        })
        .collect();
    Some((m, comp))
}

// ---------------------------------------------------------------- limpieza y filtro

/// Los subgrupos con menos de `n_min` espigas se reparten entre los válidos
/// por voto de sus 10 vecinas (contra una muestra de hasta 2 000 espigas
/// válidas). `None` si quedan menos de dos válidos.
fn limpia(sub: Vec<usize>, x: &[Vec<f64>], n_min: usize) -> Option<Vec<usize>> {
    let k = sub.iter().max().map_or(0, |&s| s + 1);
    let mut cuenta = vec![0usize; k];
    sub.iter().for_each(|&s| cuenta[s] += 1);
    let validos: Vec<usize> = (0..k).filter(|&c| cuenta[c] >= n_min).collect();
    if validos.len() < 2 {
        return None;
    }
    let es_valido = |s: usize| validos.contains(&s);
    let refs: Vec<usize> = (0..x.len()).filter(|&i| es_valido(sub[i])).collect();
    let paso = refs.len().div_ceil(2_000).max(1);
    let muestra: Vec<usize> = refs.iter().copied().step_by(paso).collect();
    let mut out = sub.clone();
    for i in 0..x.len() {
        if es_valido(sub[i]) {
            continue;
        }
        let mut d: Vec<(f64, usize)> = muestra
            .iter()
            .map(|&j| (x[i].iter().zip(&x[j]).map(|(a, b)| (a - b) * (a - b)).sum(), sub[j]))
            .collect();
        let kk = 10.min(d.len());
        d.select_nth_unstable_by(kk - 1, |a, b| a.0.total_cmp(&b.0));
        let mut votos = vec![0usize; k];
        d[..kk].iter().for_each(|&(_, s)| votos[s] += 1);
        out[i] = (0..k).max_by_key(|&c| (votos[c], cuenta[c])).unwrap();
    }
    Some(renumera(&out))
}

fn renumera(sub: &[usize]) -> Vec<usize> {
    let mut mapa = std::collections::BTreeMap::new();
    for &s in sub {
        let n = mapa.len();
        mapa.entry(s).or_insert(n);
    }
    sub.iter().map(|s| mapa[s]).collect()
}

/// Coexistencia de dos perfiles temporales: intersección de sus histogramas
/// normalizados (1 = disparan igual en el tiempo, 0 = tramos disjuntos).
pub fn coexistencia(ta: &[u64], tb: &[u64], tramo: (u64, u64), tramos: usize) -> f64 {
    let perfil = |t: &[u64]| {
        let mut h = vec![0.0; tramos];
        let ancho = (tramo.1.saturating_sub(tramo.0)).max(1) as f64;
        for &v in t {
            let b = ((v.saturating_sub(tramo.0)) as f64 / ancho * tramos as f64) as usize;
            h[b.min(tramos - 1)] += 1.0 / t.len() as f64;
        }
        h
    };
    let (ha, hb) = (perfil(ta), perfil(tb));
    ha.iter().zip(&hb).map(|(a, b)| a.min(*b)).sum()
}

/// Junta (por componentes conexas) los subgrupos que no coexisten en el tiempo.
fn junta_por_tiempo(sub: Vec<usize>, tiempos: &[u64], tramo: (u64, u64), p: &ParamsSeparacion) -> Vec<usize> {
    let k = sub.iter().max().map_or(0, |&s| s + 1);
    let por: Vec<Vec<u64>> = (0..k).map(|c| sub.iter().zip(tiempos).filter(|(s, _)| **s == c).map(|(_, &t)| t).collect()).collect();
    let mut padre: Vec<usize> = (0..k).collect();
    fn raiz(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            i = p[i];
        }
        i
    }
    for a in 0..k {
        for b in a + 1..k {
            if coexistencia(&por[a], &por[b], tramo, p.tramos) < p.coexistencia_minima {
                let (ra, rb) = (raiz(&mut padre, a), raiz(&mut padre, b));
                padre[rb] = ra;
            }
        }
    }
    let raices: Vec<usize> = sub.iter().map(|&s| raiz(&mut padre, s)).collect();
    renumera(&raices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn jacobi_reconstruye() {
        let a = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 1.0]];
        let (l, v) = jacobi(a.clone());
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| v[i][k] * l[k] * v[j][k]).sum();
                assert!((r - a[i][j]).abs() < 1e-9, "({i},{j}) {r} vs {}", a[i][j]);
            }
        }
    }

    /// Formas de onda sintéticas: plantilla + ruido; `t` es su instante.
    fn neurona(rng: &mut StdRng, plantilla: &[f32], n: usize, sd: f32, t: impl Fn(usize) -> u64) -> Vec<(Vec<f32>, u64)> {
        let g = Normal::new(0.0, sd).unwrap();
        (0..n).map(|i| (plantilla.iter().map(|&v| v + g.sample(rng)).collect(), t(i))).collect()
    }

    fn plantilla(amp: f32, ancho: f32) -> Vec<f32> {
        (0..32).map(|i| -amp * (-((i as f32 - 10.0) / ancho).powi(2)).exp()).collect()
    }

    fn corre(datos: &[(Vec<f32>, u64)]) -> Option<Vec<usize>> {
        let formas: Vec<&[f32]> = datos.iter().map(|d| d.0.as_slice()).collect();
        let t: Vec<u64> = datos.iter().map(|d| d.1).collect();
        let tramo = (*t.iter().min().unwrap(), *t.iter().max().unwrap());
        separa_unidad(&formas, &t, tramo, &ParamsSeparacion::default())
    }

    #[test]
    fn separa_dos_neuronas_solapadas_que_coexisten() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut d = neurona(&mut rng, &plantilla(1.0, 2.0), 1500, 0.15, |i| (i * 2) as u64);
        d.extend(neurona(&mut rng, &plantilla(1.3, 3.0), 1000, 0.15, |i| (i * 3) as u64));
        let sub = corre(&d).expect("debería partir");
        let a = sub[0];
        let pureza_a = sub[..1500].iter().filter(|&&s| s == a).count() as f64 / 1500.0;
        let pureza_b = sub[1500..].iter().filter(|&&s| s != a).count() as f64 / 1000.0;
        assert!(pureza_a > 0.9 && pureza_b > 0.9, "{pureza_a} {pureza_b}");
    }

    #[test]
    fn el_modelo_clasifica_espigas_nuevas() {
        let mut rng = StdRng::seed_from_u64(4);
        let (pa, pb) = (plantilla(1.0, 2.0), plantilla(1.3, 3.0));
        let mut d = neurona(&mut rng, &pa, 1500, 0.15, |i| (i * 2) as u64);
        d.extend(neurona(&mut rng, &pb, 1000, 0.15, |i| (i * 3) as u64));
        let formas: Vec<&[f32]> = d.iter().map(|x| x.0.as_slice()).collect();
        let t: Vec<u64> = d.iter().map(|x| x.1).collect();
        let (modelo, sub) = ajusta_modelo(&formas, &t, (0, 3000), &ParamsSeparacion::default()).expect("debería partir");
        assert_eq!(modelo.n_subgrupos(), 2);
        let (sa, sb) = (sub[0], sub[1500]);
        assert_ne!(sa, sb);
        let nuevas_a = neurona(&mut rng, &pa, 300, 0.15, |i| i as u64);
        let nuevas_b = neurona(&mut rng, &pb, 300, 0.15, |i| i as u64);
        let ok_a = nuevas_a.iter().filter(|x| modelo.clasifica(&x.0) == sa).count();
        let ok_b = nuevas_b.iter().filter(|x| modelo.clasifica(&x.0) == sb).count();
        assert!(ok_a > 270 && ok_b > 270, "{ok_a} {ok_b}");
    }

    #[test]
    fn no_separa_una_neurona_que_deriva() {
        // La amplitud baja de 1.4 a 0.8 a lo largo de la sesión: una nube
        // alargada que el BIC parte, y el filtro temporal vuelve a juntar.
        let mut rng = StdRng::seed_from_u64(2);
        let g = Normal::new(0.0, 0.1f32).unwrap();
        let d: Vec<(Vec<f32>, u64)> = (0..3000)
            .map(|i| {
                let amp = 1.4 - 0.6 * i as f32 / 3000.0;
                (plantilla(amp, 2.5).iter().map(|&v| v + g.sample(&mut rng)).collect(), i as u64)
            })
            .collect();
        assert_eq!(corre(&d), None);
    }

    #[test]
    fn no_separa_una_sola_neurona_estable() {
        let mut rng = StdRng::seed_from_u64(3);
        let d = neurona(&mut rng, &plantilla(1.0, 2.0), 2500, 0.15, |i| i as u64);
        assert_eq!(corre(&d), None);
    }
}
