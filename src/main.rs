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

//Tamaño de entrenamiento y lotes, deben poder ser puestas por el usuario
const TRAIN_SIZE: usize = 400;
const BATCH_SIZE: usize = 200;

//Backend de la red neuronal, ncesito ver si esto puede ser optimizable
type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

//Normaliza un vector de vectores (data) entre cero y 1 
//con el mínimo (min) y máximo (max) de los datos
fn normalize(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let min = data.iter()
        .flatten()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let max = data.iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = max - min;

    data.iter()
        .map(|fila| {
            fila.iter()
                .map(|&x| {
                    if range == 0.0 {
                        0.0 // evita división por cero si todos los valores son iguales
                    } else {
                        (x - min) / range
                    }
                })
                .collect()
        })
        .collect()
}



fn main() -> Result<(), Box<dyn Error>> {

    let file = "/home/oce/Documentos/proy/umap-mcqueen/data/Waveforms.csv"; // Archivo de ondas

    //Este es el buffer que servirá para almacenar lotes
    let mut buffer_crudo: Vec<Vec<f64>> = Vec::new(); // para datos crudos

    //Configuración UMAP
    let config = UmapConfig {
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
    };
    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);

    //Configuración DenStream
    let mut ds = DenStream::new(0.1, 3)
        .with_beta(0.5)
        .with_lambda(0.01)
        .with_mu(1.0);

    //Lector de CSV
    let archiv = File::open(file)?;
    let mut lector = ReaderBuilder::new()
        .from_reader(archiv);
    let mut records = lector.records().enumerate();

    //Datos para entrenamiento
    for (i, result) in &mut records {
        let registro = result?; // Toma los datos del csv y los pasa a un string

        let fila: Vec<f64> = registro // Formateamos la fila en un vector
            .iter()
            .map(|campo| campo.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;

        buffer_crudo.push(fila); // Añadimos el nuevo vector al buffer


        if buffer_crudo.len() == TRAIN_SIZE {
            println!("Entrenamiento en proceso con {} vectores", i);

            break; // Así, buffer es primer lote de vectores de entrenamiento.
        }
    }

    //Reducimos el encaje primario y a la par entrenamos
    let fitted = umap.fit(buffer_crudo.clone(), None); // UMAPEAR
    let encaje = fitted.embedding();
    println!("Dimensión reducida del encaje primario: {} × {}", encaje.len(), encaje[0].len());

    buffer_crudo.clear();

    let mut total_puntos: Vec<(f32, f32)> = Vec::new(); 

    for (i, result) in records { // seguimos con el archivo
        let registro = result?;
        let fila: Vec<f64> = registro
            .iter()
            .map(|campo| campo.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;

        buffer_crudo.push(fila);

        if buffer_crudo.len() == BATCH_SIZE {
            let nuevo_embedding = fitted.transform(buffer_crudo.clone()); // Nuevo embedding UMAP
            let embedding_normalizado = normalize(nuevo_embedding);

            //Tenemos que normalizar los datos de salida, esto para que 
            //DenStream trabaje bajo los parametros dados.
            let puntos_f32: Vec<Vec<f32>> = embedding_normalizado
                .iter()
                .map(|fila| vec![fila[0] as f32, fila[1] as f32])
                .collect();

            let _ = ds.update_batch(&puntos_f32);

            let n_clusters = ds.n_clusters();

            println!("--- Lote en índice {}: {} micro clusters ---", i, n_clusters);

            let puntos: Vec<(f32, f32)> = embedding_normalizado // Para graficar el embedding, 
                .iter()
                .map(|fila| (fila[0] as f32, fila[1] as f32))
                .collect();

            total_puntos.extend(puntos.clone());

            println!("--- Embedding en el índice {} ---", i);
            Chart::new_with_y_range(230, 120, 0.0, 1.0, 0.0, 1.0)
                .lineplot(&Shape::Points(&puntos))
                .display();

            buffer_crudo.clear();
        }
    }

    Ok(())
}


