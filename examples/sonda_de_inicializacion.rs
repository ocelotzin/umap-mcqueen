//! ¿Los pesos iniciales ya difieren, o el no determinismo nace en el bucle?
//!
//! Bisección. Tras descartar autotune, el GPU, las reducciones paralelas, el azar
//! sin sembrar de fast-umap, los HashMap, el timeout por reloj y que `Autodiff` no
//! reenviara la semilla, queda saber en qué punto empieza la divergencia.
//!
//! Se crean DOS modelos sin entrenar, resembrando antes de cada uno igual que hace
//! `fit_with_signal`, y se comparan proyectando el mismo dato.
//!
//!   - Si coinciden  → la inicialización es determinista y el problema está en el
//!                     bucle de entrenamiento.
//!   - Si difieren   → es la inicialización de pesos, y `B::seed` no basta.

use burn::tensor::backend::Backend as _;
use fast_umap::model::{UMAPModel, UMAPModelConfigBuilder};
use std::error::Error;

use cubecl::wgpu::WgpuRuntime;
type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Default::default();
    let cfg = UMAPModelConfigBuilder::default()
        .input_size(48)
        .hidden_sizes(vec![128])
        .output_size(2)
        .build()?;

    // Un dato fijo con el que sondear la red sin entrenar.
    let dato: Vec<f32> = (0..48).map(|i| (i as f32 * 0.137).sin()).collect();

    let mut salidas = Vec::new();
    for _ in 0..3 {
        // Exactamente lo que hace `fit_with_signal` antes de crear el modelo.
        <MyAutodiffBackend as burn::tensor::backend::Backend>::seed(&device, 9999);
        let m: UMAPModel<MyBackend> = UMAPModel::new(&cfg, &device);
        let x = burn::tensor::Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(dato.clone(), [1, 48]), &device);
        let y = m.forward(x);
        salidas.push(y.into_data().to_vec::<f32>().unwrap());
    }

    println!("proyección del mismo dato por 3 modelos recién creados:");
    for (i, s) in salidas.iter().enumerate() {
        println!("   modelo {i}: {s:?}");
    }
    let iguales = salidas.windows(2).all(|p| p[0] == p[1]);
    println!("\n→ inicialización de pesos: {}",
             if iguales { "DETERMINISTA — el problema está en el bucle de entrenamiento" }
             else { "NO determinista — `B::seed` no alcanza a la creación del modelo" });
    Ok(())
}
