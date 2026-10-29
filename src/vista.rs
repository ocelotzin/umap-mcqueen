//! Lo que la CLI muestra de las etiquetas: proporciones, encaje coloreado y
//! raster. Un color por etiqueta, el mismo en todas partes; el ruido en gris.

use std::collections::BTreeMap;

use rgb::RGB8;
use textplots::{Chart, ColorPlot, Shape};
use umap_mcqueen::denstream::ETIQUETA_RUIDO;


/// Colores por etiqueta: los mismos en la gráfica y en las tablas.
pub const PALETA: [RGB8; 10] = [
    RGB8 { r: 230, g: 25, b: 75 },   // rojo
    RGB8 { r: 60, g: 180, b: 75 },   // verde
    RGB8 { r: 0, g: 130, b: 200 },   // azul
    RGB8 { r: 245, g: 130, b: 48 },  // naranja
    RGB8 { r: 145, g: 30, b: 180 },  // morado
    RGB8 { r: 70, g: 240, b: 240 },  // cian
    RGB8 { r: 240, g: 50, b: 230 },  // magenta
    RGB8 { r: 210, g: 245, b: 60 },  // lima
    RGB8 { r: 255, g: 225, b: 25 },  // amarillo
    RGB8 { r: 170, g: 110, b: 40 },  // marrón
];
pub const GRIS: RGB8 = RGB8 { r: 128, g: 128, b: 128 };

pub fn color_de(etiqueta: i64) -> RGB8 {
    if etiqueta < 0 { GRIS } else { PALETA[etiqueta as usize % PALETA.len()] }
}

/// Texto en el color de la etiqueta.
///
/// Con `colored`, la MISMA biblioteca (y versión) que usa textplots para el
/// encaje: así tablas, raster y gráfica deciden igual si colorear (sí en una
/// terminal, no al redirigir a un archivo; `NO_COLOR` lo apaga y
/// `CLICOLOR_FORCE=1` lo fuerza) y degradan igual a 8 colores cuando la
/// terminal no anuncia color verdadero (`COLORTERM=truecolor`).
pub fn pinta(etiqueta: i64, texto: &str) -> String {
    use colored::Colorize;
    let c = color_de(etiqueta);
    texto.truecolor(c.r, c.g, c.b).to_string()
}

pub fn nombre(etiqueta: i64) -> String {
    if etiqueta == ETIQUETA_RUIDO { "ruido".to_string() } else { etiqueta.to_string() }
}

/// Etiquetas distintas, sin contar el ruido.
pub fn n_etiquetas(conteo: &BTreeMap<i64, usize>) -> usize {
    conteo.keys().filter(|&&e| e >= 0).count()
}

/// Etiquetas de mayor a menor, con el ruido al final.
pub fn ordenadas(conteo: &BTreeMap<i64, usize>) -> Vec<(i64, usize)> {
    let mut v: Vec<(i64, usize)> = conteo.iter().filter(|(e, _)| **e >= 0).map(|(&e, &n)| (e, n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if let Some(&n) = conteo.get(&ETIQUETA_RUIDO) {
        v.push((ETIQUETA_RUIDO, n));
    }
    v
}

/// Una línea por lote: "  3: 41.0% · 0: 28.5% · ruido: 2.0%", y debajo una barra
/// apilada de 60 caracteres con la misma proporción y colores.
pub fn proporciones_en_linea(conteo: &BTreeMap<i64, usize>) -> String {
    let total: usize = conteo.values().sum();
    if total == 0 {
        return "    (lote vacío)".to_string();
    }
    let v = ordenadas(conteo);
    let texto = v
        .iter()
        .map(|&(e, n)| pinta(e, &format!("{}: {:.1}%", nombre(e), 100.0 * n as f64 / total as f64)))
        .collect::<Vec<_>>()
        .join(" · ");
    format!("    {}\n    {}", texto, barra_apilada(&v, total, 60))
}

/// Barra horizontal dividida en tramos proporcionales, uno por etiqueta.
pub fn barra_apilada(v: &[(i64, usize)], total: usize, ancho: usize) -> String {
    let mut barra = String::new();
    let mut usados = 0usize;
    let mut acumulado = 0usize;
    for &(e, n) in v {
        acumulado += n;
        //Redondeo sobre el acumulado: los tramos siempre suman `ancho`.
        let hasta = (acumulado * ancho + total / 2) / total;
        let tramo = hasta.saturating_sub(usados);
        usados += tramo;
        barra.push_str(&pinta(e, &"█".repeat(tramo)));
    }
    barra
}

/// Tabla final: formas de onda y proporción por etiqueta, con barra.
pub fn imprime_proporciones(conteo: &BTreeMap<i64, usize>) {
    let total: usize = conteo.values().sum();
    if total == 0 {
        println!("No se etiquetó ninguna forma de onda.");
        return;
    }
    let v = ordenadas(conteo);
    let max = v.iter().map(|&(_, n)| n).max().unwrap_or(1);
    println!("{:>9} {:>9} {:>8}", "etiqueta", "formas", "%");
    for &(e, n) in &v {
        let largo = ((n * 40 + max / 2) / max).max(1);
        println!(
            "{} {:>9} {:>7.1}%  {}",
            pinta(e, &format!("{:>9}", nombre(e))),
            n,
            100.0 * n as f64 / total as f64,
            pinta(e, &"█".repeat(largo))
        );
    }
    println!("{:>9} {:>9}", "total", total);
    println!("\n{}", barra_apilada(&v, total, 80));
}

// --------------------------------------------------------------- raster

/// Ancho visible de un texto, sin contar los códigos ANSI de color.
fn ancho_visible(texto: &str) -> usize {
    let mut n = 0;
    let mut en_escape = false;
    for ch in texto.chars() {
        match (en_escape, ch) {
            (false, '\x1b') => en_escape = true,
            (true, 'm') => en_escape = false,
            (true, _) => {}
            (false, _) => n += 1,
        }
    }
    n
}

/// Pone dos bloques de texto uno al lado del otro, alineados arriba.
pub fn lado_a_lado(izquierda: &str, derecha: &[String], separacion: usize) -> String {
    let izq: Vec<&str> = izquierda.lines().collect();
    let ancho = izq.iter().map(|l| ancho_visible(l)).max().unwrap_or(0);
    let filas = izq.len().max(derecha.len());
    let mut salida = String::new();
    for f in 0..filas {
        let l = izq.get(f).copied().unwrap_or("");
        salida.push_str(l);
        if let Some(r) = derecha.get(f) {
            salida.push_str(&" ".repeat(ancho - ancho_visible(l) + separacion));
            salida.push_str(r);
        }
        salida.push('\n');
    }
    salida
}

/// El encaje normalizado, un color por etiqueta, como texto (para poder
/// ponerle el raster al lado). `ancho_px`×`alto_px` en puntos braille: cada
/// carácter son 2×4 puntos.
pub fn encaje_coloreado(puntos: &[(f32, f32)], etiquetas: &[i64], ancho_px: u32, alto_px: u32) -> String {
    let mut por_etiqueta: BTreeMap<i64, Vec<(f32, f32)>> = BTreeMap::new();
    for (p, &e) in puntos.iter().zip(etiquetas) {
        por_etiqueta.entry(e).or_default().push(*p);
    }
    let formas: Vec<(Shape, RGB8)> = por_etiqueta
        .iter()
        .map(|(&e, v)| (Shape::Points(v), color_de(e)))
        .collect();
    let mut grafica = Chart::new_with_y_range(ancho_px, alto_px, 0.0, 1.0, 0.0, 1.0);
    let mut g = &mut grafica;
    for (forma, color) in &formas {
        g = g.linecolorplot(forma, *color);
    }
    g.axis();
    g.figures();
    format!("{g}")
}

/// Filas del raster que caben: las etiquetas con más formas de onda, y el
/// resto juntas en «otras». Siempre en orden de etiqueta, con el ruido al final,
/// para que una unidad ocupe la misma fila de un lote al siguiente.
fn filas_del_raster(conteo: &BTreeMap<i64, usize>, max_filas: usize) -> (Vec<i64>, bool) {
    let mut etiquetas: Vec<(i64, usize)> =
        conteo.iter().filter(|(e, _)| **e >= 0).map(|(&e, &n)| (e, n)).collect();
    let hay_ruido = conteo.contains_key(&ETIQUETA_RUIDO);
    let cupo = max_filas.saturating_sub(hay_ruido as usize).max(1);
    let otras = etiquetas.len() > cupo;
    if otras {
        etiquetas.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        etiquetas.truncate(cupo - 1);
    }
    let mut filas: Vec<i64> = etiquetas.into_iter().map(|(e, _)| e).collect();
    filas.sort_unstable();
    if hay_ruido {
        filas.push(ETIQUETA_RUIDO);
    }
    (filas, otras)
}

/// Raster de UN lote: una fila por etiqueta y, en horizontal, el orden de
/// llegada de las formas de onda (el tiempo, a falta de marcas de tiempo en el
/// CSV). Cada `│` es una forma de onda de esa etiqueta; `┃` si en esa columna
/// caen varias. Sin color, el rótulo de cada fila dice qué etiqueta es. `primera` es el índice global de la primera forma del lote.
pub fn raster_lote(etiquetas: &[i64], primera: usize, ancho: usize, max_filas: usize) -> Vec<String> {
    let n = etiquetas.len();
    let mut conteo: BTreeMap<i64, usize> = BTreeMap::new();
    for &e in etiquetas {
        *conteo.entry(e).or_default() += 1;
    }
    let (filas, hay_otras) = filas_del_raster(&conteo, max_filas);
    const OTRAS: i64 = i64::MIN;
    let fila_de = |e: i64| -> i64 { if filas.contains(&e) { e } else { OTRAS } };

    // celdas[fila][columna] = formas de onda que caen ahí
    let mut orden: Vec<i64> = filas.clone();
    if hay_otras {
        let pos = orden.iter().position(|&e| e == ETIQUETA_RUIDO).unwrap_or(orden.len());
        orden.insert(pos, OTRAS);
    }
    let mut celdas = vec![vec![0u16; ancho]; orden.len()];
    for (k, &e) in etiquetas.iter().enumerate() {
        let col = if n <= 1 { 0 } else { k * ancho / n };
        let f = orden.iter().position(|&x| x == fila_de(e)).unwrap();
        celdas[f][col.min(ancho - 1)] += 1;
    }

    let mut lineas = vec![format!("raster del lote · {n} formas de onda en orden de llegada")];
    for (f, &e) in orden.iter().enumerate() {
        let rotulo = if e == OTRAS { "otras".to_string() } else { nombre(e) };
        let total: usize = celdas[f].iter().map(|&c| c as usize).sum();
        let traza: String = celdas[f]
            .iter()
            .map(|&c| match c { 0 => ' ', 1 => '│', _ => '┃' })
            .collect();
        let color = if e == OTRAS { ETIQUETA_RUIDO } else { e };
        lineas.push(format!(
            "{} {}{}{} {:>4}",
            pinta(color, &format!("{rotulo:>6}")),
            "▕",
            pinta(color, &traza),
            "▏",
            total
        ));
    }
    let ultima = primera + n.saturating_sub(1);
    let izq = primera.to_string();
    let der = ultima.to_string();
    lineas.push(format!(
        "{:>6}  {}{}{}",
        "",
        izq,
        " ".repeat((ancho + 2).saturating_sub(izq.len() + der.len())),
        der
    ));
    lineas
}

/// Raster de la SESIÓN: una fila por etiqueta y una columna por lote (o por
/// grupo de lotes, si no caben). La intensidad del bloque es la fracción del
/// lote que se llevó esa etiqueta: así se ve cuándo nace una unidad, cuándo
/// deja de aparecer y cómo cambian las proporciones.
pub fn raster_sesion(historial: &[BTreeMap<i64, usize>], ancho: usize, max_filas: usize) -> String {
    if historial.is_empty() {
        return String::new();
    }
    let lotes_por_col = historial.len().div_ceil(ancho).max(1);
    let columnas: Vec<BTreeMap<i64, usize>> = historial
        .chunks(lotes_por_col)
        .map(|grupo| {
            let mut m = BTreeMap::new();
            for c in grupo {
                for (&e, &n) in c {
                    *m.entry(e).or_insert(0usize) += n;
                }
            }
            m
        })
        .collect();
    let mut total: BTreeMap<i64, usize> = BTreeMap::new();
    for c in &columnas {
        for (&e, &n) in c {
            *total.entry(e).or_default() += n;
        }
    }
    let (filas, hay_otras) = filas_del_raster(&total, max_filas);

    let bloque = |frac: f64| -> char {
        match frac {
            f if f <= 0.0 => ' ',
            f if f < 0.05 => '·',
            f if f < 0.15 => '░',
            f if f < 0.35 => '▒',
            f if f < 0.6 => '▓',
            _ => '█',
        }
    };
    let mut salida = format!(
        "Raster de la sesión · {} lotes{} · intensidad = fracción del lote ( · <5%  ░ <15%  ▒ <35%  ▓ <60%  █ )\n",
        historial.len(),
        if lotes_por_col > 1 { format!(", {lotes_por_col} por columna") } else { String::new() }
    );
    let mut pinta_fila = |rotulo: String, color: i64, cuenta: &dyn Fn(&BTreeMap<i64, usize>) -> usize| {
        let traza: String = columnas
            .iter()
            .map(|c| {
                let n: usize = c.values().sum();
                bloque(if n == 0 { 0.0 } else { cuenta(c) as f64 / n as f64 })
            })
            .collect();
        salida.push_str(&format!("{} ▕{}▏\n", pinta(color, &format!("{rotulo:>6}")), pinta(color, &traza)));
    };
    for &e in filas.iter().filter(|&&e| e >= 0) {
        pinta_fila(nombre(e), e, &|c| c.get(&e).copied().unwrap_or(0));
    }
    if hay_otras {
        let visibles = filas.clone();
        pinta_fila("otras".into(), ETIQUETA_RUIDO, &|c| {
            c.iter().filter(|(e, _)| **e >= 0 && !visibles.contains(e)).map(|(_, &n)| n).sum()
        });
    }
    if filas.contains(&ETIQUETA_RUIDO) {
        pinta_fila(nombre(ETIQUETA_RUIDO), ETIQUETA_RUIDO, &|c| c.get(&ETIQUETA_RUIDO).copied().unwrap_or(0));
    }
    let n_col = columnas.len();
    let der = format!("lote {}", historial.len());
    salida.push_str(&format!(
        "{:>6}  lote 1{}{}\n",
        "",
        " ".repeat((n_col + 2).saturating_sub(6 + der.len())),
        der
    ));
    salida
}

// -------------------------------------------------------- formas de onda

/// Gris tenue para el fondo de formas de onda (más oscuro que el del ruido).
const FONDO: RGB8 = RGB8 { r: 85, g: 85, b: 90 };

/// Las formas de onda crudas del lote encimadas, voltaje contra muestra (como
/// «200 spikes al azar»), y encima la forma de onda MEDIA de cada etiqueta en
/// su color.
///
/// Las medias en vez de colorear cada trazo: en braille cada carácter tiene un
/// solo color, el del último trazo que pasa por él, y con 200 trazos encimados
/// la etiqueta que se dibuja al final tapa a todas. Las medias, de la etiqueta
/// más grande a la más pequeña, se distinguen siempre. Devuelve las líneas,
/// con un título arriba, listas para ponerlas en una columna.
pub fn formas_de_onda(lote: &[Vec<f64>], etiquetas: &[i64], ancho_px: u32, alto_px: u32) -> Vec<String> {
    let dim = lote.first().map(|f| f.len()).unwrap_or(0);
    if dim < 2 || lote.iter().any(|f| f.len() != dim) {
        return Vec::new();
    }
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for x in lote.iter().flatten().filter(|x| x.is_finite()) {
        ymin = ymin.min(*x);
        ymax = ymax.max(*x);
    }
    if !(ymin < ymax) {
        return Vec::new();
    }

    // Fondo: todas las formas de onda, en gris.
    let trazos: Vec<Vec<(f32, f32)>> = lote
        .iter()
        .map(|f| f.iter().enumerate().map(|(k, &v)| (k as f32, v as f32)).collect())
        .collect();

    // Media por etiqueta (sin el ruido), de la más grande a la más pequeña.
    let mut sumas: BTreeMap<i64, (Vec<f64>, usize)> = BTreeMap::new();
    for (f, &e) in lote.iter().zip(etiquetas).filter(|(_, e)| **e >= 0) {
        let s = sumas.entry(e).or_insert_with(|| (vec![0.0; dim], 0));
        for (a, &x) in s.0.iter_mut().zip(f) {
            *a += x;
        }
        s.1 += 1;
    }
    let mut medias: Vec<(i64, usize, Vec<(f32, f32)>)> = sumas
        .into_iter()
        .map(|(e, (s, n))| {
            let m = s.iter().enumerate().map(|(k, &x)| (k as f32, (x / n as f64) as f32)).collect();
            (e, n, m)
        })
        .collect();
    medias.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut formas: Vec<(Shape, RGB8)> = trazos.iter().map(|t| (Shape::Lines(t), FONDO)).collect();
    formas.extend(medias.iter().map(|(e, _, m)| (Shape::Lines(m), color_de(*e))));

    let mut grafica = Chart::new_with_y_range(
        ancho_px, alto_px, 0.0, (dim - 1) as f32, ymin as f32, ymax as f32);
    let mut g = &mut grafica;
    for (forma, color) in &formas {
        g = g.linecolorplot(forma, *color);
    }
    g.axis();
    g.figures();

    let mut lineas = vec![format!(
        "formas de onda del lote ({} × {dim} muestras) · en color, la media de cada etiqueta",
        lote.len()
    )];
    lineas.extend(format!("{g}").lines().map(str::to_string));
    lineas
}
