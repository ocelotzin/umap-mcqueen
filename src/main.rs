//Librerías para leer el csv
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

//Fast UMAP
use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;

//Para graficar
use textplots::{Chart, Plot, Shape};

//Para clustering
use clump::DenStream;

//Guardado y carga verificada del modelo
mod modelo;
//Normalizacion con referencia fija
mod normaliza;
use normaliza::{Normalizador, NormalizadorDeEntrada};

//Tamaño de entrenamiento y lotes, ahora puestos por el usuario desde la línea
//de órdenes (ver `Opciones`), con estos valores por defecto.
const TRAIN_SIZE: usize = 400;
const BATCH_SIZE: usize = 200;

//Backend de la red neuronal, ncesito ver si esto puede ser optimizable
type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

//Opciones de uso
struct Opciones {
    csv: String, //Ruta del dataset
    train_size: usize, // Lote de entrenamiento
    batch_size: usize, // Lote de trabajo
    modelo: Option<String>, // Pesos y datos del modelo a usar
    cargar: bool,
}

const USO: &str = "\
uso: umap-mcqueen <Waveforms.csv> [opciones]

  --modelo <ruta>   guarda el modelo entrenado en <ruta>.bin y <ruta>.json
  --cargar          carga el modelo de --modelo en vez de entrenar
  --train <n>       formas de onda de entrenamiento (por defecto 400)
  --lote <n>        tamaño de lote (por defecto 200)
";

//Argumentos de ejecución, regresa una variable tipo opciones y comprueba que
//dadas opciones sean válidas.
fn opciones() -> Result<Opciones, Box<dyn Error>> {
    let mut args = std::env::args().skip(1); //Iterador sobre los argumentos dados
    let mut o = Opciones {
        csv: String::new(),
        train_size: TRAIN_SIZE,
        batch_size: BATCH_SIZE,
        modelo: None,
        cargar: false,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--modelo" => o.modelo = Some(args.next().ok_or("--modelo necesita una ruta")?),
            "--cargar" => o.cargar = true,
            "--train" => o.train_size = args.next().ok_or("--train necesita un número")?.parse()?,
            "--lote" => o.batch_size = args.next().ok_or("--lote necesita un número")?.parse()?,
            "-h" | "--help" => {
                print!("{USO}");
                std::process::exit(0);
            }
            otro if otro.starts_with('-') => return Err(format!("opción desconocida: {otro}\n\n{USO}").into()),
            ruta => o.csv = ruta.to_string(),
        }
    }
    if o.csv.is_empty() {
        return Err(format!("falta el fichero de formas de onda\n\n{USO}").into());
    }
    if o.cargar && o.modelo.is_none() {
        return Err("--cargar necesita --modelo".into());
    }
    Ok(o)
}

//Crea un lector
fn lector(ruta: &str) -> Result<csv::Reader<File>, Box<dyn Error>> {
    Ok(ReaderBuilder::new().has_headers(false).from_reader(File::open(ruta)?))
}

//Del lctor toma un registro y lo convierte en filas
fn fila_de(registro: &csv::StringRecord) -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(registro
        .iter()
        .map(|campo| campo.trim().parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?)
}

/// La configuración de UMAP, en un sitio, para que entrenar y cargar usen la misma.
// No entiendo esta función, simplemente podríamos crear una variable con esto
// esta es una función constante.
fn configuracion() -> UmapConfig {
    UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 200,
            learning_rate: 1e-3,
            patience: Some(50),
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

// Procesa los errores en el loop principal para facilitar su lectura
// revisar si esta práctica es común y eficiente
fn main() {
    if let Err(e) = main_err() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn main_err() -> Result<(), Box<dyn Error>> {
    let opts = opciones()?;

    //Este es el buffer que servirá para almacenar lotes
    let mut buffer_crudo: Vec<Vec<f64>> = Vec::new(); // para datos crudos

    // Única vez en la que se va a usar la función que pasa la configuración,
    // por este motivo sugiero quitar esta complejidad extra, DenStream es
    // un ejemplo de por qué esta configuración es una variable mutable.
    let config = configuracion();

    //Configuración DenStream
    let mut ds = DenStream::new(0.1, 3)
        .with_beta(0.5)
        .with_lambda(0.01)
        .with_mu(1.0);

    //Lector de CSV
    let mut lector = lector(&opts.csv)?;
    let mut records = lector.records().enumerate();

    //Datos para entrenamiento
    for (i, result) in &mut records {
        let registro = result?; // Toma los datos del csv y los pasa a un string
        buffer_crudo.push(fila_de(&registro)?); // Añadimos el nuevo vector al buffer

        if buffer_crudo.len() == opts.train_size {
            println!("Entrenamiento en proceso con {} vectores", i);
            break; // Así, buffer es primer lote de vectores de entrenamiento.
        }
    }
    if buffer_crudo.len() < opts.train_size {
        return Err(format!(
            "{} sólo tiene {} formas de onda y se pidieron {} de entrenamiento",
            opts.csv, buffer_crudo.len(), opts.train_size
        )
        .into());
    }
    let n_dim = buffer_crudo[0].len();

    //O cargamos un modelo ya entrenado, o entrenamos uno nuevo.
    let (fitted, normalizador, normalizador_de_entrada) = if opts.cargar {
        let base = opts.modelo.as_ref().expect("--cargar exige --modelo");
        let (m, manifiesto) = modelo::carga_verificada::<MyAutodiffBackend>(
            base,
            config.clone(),
            n_dim,
            Default::default(),
        )?;
        println!(
            "Modelo cargado de {base}.bin — entrenado con {} vectores, huella {}…",
            manifiesto.n_entrenamiento,
            &manifiesto.huella_datos[..12]
        );
        // Esto suena mucho a ia, y es comportamiento esperado en la documentación
        println!(
            "⚠ El encaje de entrenamiento NO se guarda: `embedding()` de un modelo \
             cargado está vacío por construcción, no por error."
        );
        let norm_entrada = manifiesto.normalizacion_de_entrada.clone().ok_or(
            "el manifiesto no trae referencia de normalización de ENTRADA. Sin ella, \
             `transform` recibe datos crudos cuando la red se entrenó con datos \
             normalizados, y devuelve coordenadas cinco órdenes de magnitud fuera \
             del encaje (ver normaliza::NormalizadorDeEntrada).",
        )?;
        let norm = manifiesto.normalizacion.ok_or(
            "el manifiesto no trae referencia de normalización (¿formato 1?). Sin ella \
             las coordenadas de esta sesión no son comparables con las de la sesión que \
             entrenó el modelo.",
        )?;
        (m, norm, norm_entrada)
    } else {
        //Reducimos el encaje primario y a la par entrenamos
        let m = fast_umap::Umap::<MyAutodiffBackend>::new(config.clone()).fit(buffer_crudo.clone(), None); // UMAPEAR
        //La red se entrena sobre datos normalizados por caracteristica; hay que
        //quedarse con ESA referencia para poder aplicarla luego en `transform`.
        let norm_entrada = NormalizadorDeEntrada::ajusta(&buffer_crudo).ok_or(
            "no se pudo medir la normalización de entrada sobre el lote de entrenamiento",
        )?;
        let encaje = m.embedding();
        println!("Dimensión reducida del encaje primario: {} × {}", encaje.len(), encaje[0].len());

        //La referencia de normalización se mide UNA VEZ, aquí, sobre el encaje de
        //entrenamiento. Antes se medía por lote, y eso movía el suelo bajo DenStream
        //más que su propio epsilon (ver `normaliza.rs`).
        let norm = Normalizador::ajusta(encaje).ok_or(
            "no se pudo medir la referencia de normalización sobre el encaje de \
             entrenamiento (¿vacío, o todos los valores iguales?)",
        )?;
        println!("Referencia de normalización fijada: [{:.4}, {:.4}]", norm.min, norm.max);

        if let Some(base) = &opts.modelo {
            let rutas = modelo::guarda(&m, &config, &buffer_crudo, Some(norm), Some(norm_entrada.clone()), base)?;
            println!("Modelo guardado:");
            println!("  pesos      : {}", rutas.pesos.display());
            println!("  manifiesto : {}", rutas.manifiesto.display());
        }
        (m, norm, norm_entrada)
    };

    buffer_crudo.clear();

    let mut total_puntos: Vec<(f32, f32)> = Vec::new();

    //Procesa un lote: encaje, normalización, DenStream y gráfica.
    let procesa_lote = |lote: &[Vec<f64>], i: usize, ds: &mut DenStream, total: &mut Vec<(f32, f32)>| {
        //`fast-umap` NO normaliza en `transform`, aunque sí lo hace al entrenar.
        //Sin esta línea la red recibe datos crudos y devuelve coordenadas cinco
        //órdenes de magnitud fuera del encaje.
        let entrada = normalizador_de_entrada
            .aplica(lote.to_vec())
            .expect("la dimensión del lote no corresponde al modelo");
        let nuevo_embedding = fitted.transform(entrada); // Nuevo embedding UMAP
        //Referencia FIJA, la del entrenamiento: es lo que hace que un punto caiga
        //siempre en la misma coordenada, llegue en el lote que llegue.
        let embedding_normalizado = normalizador.aplica(nuevo_embedding);
        let fuera = normalizador.fuera_de_rango(&embedding_normalizado);

        //Tenemos que normalizar los datos de salida, esto para que
        //DenStream trabaje bajo los parametros dados.
        let puntos_f32: Vec<Vec<f32>> = embedding_normalizado
            .iter()
            .map(|fila| vec![fila[0] as f32, fila[1] as f32])
            .collect();

        let _ = ds.update_batch(&puntos_f32);

        //Con referencia fija, salirse de [0,1] deja de ser imposible y pasa a ser
        //una señal: si crece, el modelo ya no representa lo que llega.
        println!(
            "--- Lote en índice {}: {} micro clusters · {:.1}% fuera de [0,1] ---",
            i, ds.n_clusters(), 100.0 * fuera
        );

        let puntos: Vec<(f32, f32)> = embedding_normalizado // Para graficar el embedding,
            .iter()
            .map(|fila| (fila[0] as f32, fila[1] as f32))
            .collect();

        total.extend(puntos.clone());

        println!("--- Embedding en el índice {} ---", i);
        Chart::new_with_y_range(230, 120, 0.0, 1.0, 0.0, 1.0)
            .lineplot(&Shape::Points(&puntos))
            .display();
    };

    let mut ultimo = 0usize;
    for (i, result) in records { // seguimos con el archivo
        let registro = result?;
        buffer_crudo.push(fila_de(&registro)?);
        ultimo = i;

        if buffer_crudo.len() == opts.batch_size {
            procesa_lote(&buffer_crudo, i, &mut ds, &mut total_puntos);
            buffer_crudo.clear();
        }
    }

    //El último lote incompleto se descartaba en silencio: hasta `batch_size - 1`
    //formas de onda del final del fichero no se procesaban nunca.
    if !buffer_crudo.is_empty() {
        println!("--- Último lote, incompleto: {} formas ---", buffer_crudo.len());
        procesa_lote(&buffer_crudo, ultimo, &mut ds, &mut total_puntos);
    }

    println!("\nTotal de puntos procesados: {}", total_puntos.len());

    Ok(())
}
