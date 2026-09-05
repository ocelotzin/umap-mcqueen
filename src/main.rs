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
use clump::Hdbscan;

//Tamaño de entrenamiento y lotes, deben poder ser puestas por el usuario
const TRAIN_SIZE: usize = 400;
const BATCH_SIZE: usize = 150;

//Backend de la red neuronal, ncesito ver si esto puede ser optimizable
type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn main() -> Result<(), Box<dyn Error>> {

    let file = "/home/oce/proy/umap-mcqueen/data/Waveforms.csv"; // Archivo de ondas

    //Este es el buffer que servirá para almacenar lotes
    let mut buffer: Vec<Vec<f64>> = Vec::new(); // vector de vectores
    
    //Configuración UMAP
    let config = UmapConfig::default();
    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);

    //Lector de CSV
    let archiv = File::open(file)?;
    let mut lector = ReaderBuilder::new()
        .from_reader(archiv);
    let mut records = lector.records().enumerate();

    for (i, result) in &mut records {
        let registro = result?; // Toma los datos del csv y los pasa a un string

        let fila: Vec<f64> = registro // Formateamos la fila en un vector
            .iter()
            .map(|campo| campo.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;

        buffer.push(fila); // Añadimos el nuevo vector al buffer


        if buffer.len() == TRAIN_SIZE {
            println!("Entrenamiento en proceso con {} vectores", i);
            break; // Así, buffer es primer lote de vectores de entrenamiento.
        }
    }

    //Crear encaje primario
    let fitted = umap.fit(buffer.clone(), None); // UMAPEAR
    let encaje = fitted.embedding();
    println!("Dimensión reducida del encaje primario: {} × {}", encaje.len(), encaje[0].len());
                                   
    buffer.clear();

    for (i, result) in records { // seguimos con el archivo
        let registro = result?;
        let fila: Vec<f64> = registro
            .iter()
            .map(|campo| campo.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;

        buffer.push(fila);

        if buffer.len() == BATCH_SIZE {
            let nuevo_embedding = fitted.transform(buffer.clone()); // Nuevo embedding parametrico
  

            let puntos_f32: Vec<Vec<f32>> = nuevo_embedding
                .iter()
                .map(|fila| vec![fila[0] as f32, fila[1] as f32])
                .collect();

            // HDBSCAN corre desde cero en cada lote (no es streaming)
            let hdbscan = Hdbscan::new().with_min_samples(2).with_min_cluster_size(2);
            let labels = hdbscan.fit_predict(&puntos_f32)?;

            // Número de clusters: etiqueta máxima + 1, ignorando ruido (usize::MAX)
            let n_clusters = labels
                .iter()
                .filter(|&&l| l != usize::MAX)
                .copied()
                .max()
                .map_or(0, |m| m.saturating_add(1));

            println!("--- Lote en índice {}: {} clusters ---", i, n_clusters);

            let puntos: Vec<(f32, f32)> = nuevo_embedding // Para graficar el embedding, 
                .iter()
                .map(|fila| (fila[0] as f32, fila[1] as f32))
                .collect();

            let xs: Vec<f32> = puntos.iter().map(|p| p.0).collect();
            let xmin = xs.iter().cloned().fold(f32::INFINITY, f32::min);
            let xmax = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);


            println!("--- Embedding en el índice {} ---", i);
            Chart::new(120, 80, xmin, xmax)
                .lineplot(&Shape::Points(&puntos))
                .display();

            buffer.clear();
        }
    }

    Ok(())
}


