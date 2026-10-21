# umap-mcqueen

Reducción de dimensiones a tiempo real y *offline* por medio de UMAP e implementado en Rust para mayor eficiencia en recursos y tiempo.

----

El algoritmo usado es ' fast-umap ', una implementación de UMAP en Rust que a parte de ser mucho más veloz, es paramétrica, por lo que rompe con la limitación usual de UMAP que es no poder aceptar nuevos datos en base a un encaje ya realizado.

Después de ser reducidos de dimensionalidad, se usa un algoritmo de clustering usual.

![gráfica de UMAP](img/graf-terminal.png)

Su única funcionalidad actual es tomar un archivo ' .csv ' e iterar sobre sus datos sólo aplicando UMAP por lotes. Lo siguiente es TODO:

- Realizar el clustering final.
- Mostrar a color las clasificaciones.
- Implementar la lectura a tiempo real de un *web socket*.

-----

## Características deseables

- Codigo mínimo.
- Funcionalidades mínimas.
- Adaptable a otros módulos.

-----

## Compilar desde un clon limpio

Hace falta `cargo` ≥ 1.85 (`edition = "2024"`; con `rustup` basta) y, para **correr**, una GPU con Vulkan: el
backend es `wgpu`. Compilar no la necesita.

```bash
git clone https://github.com/ocelotzin/umap-mcqueen && cd umap-mcqueen
cargo build --release      # trae fast-umap del fork de abajo, por git
cargo test                 # las pruebas de normaliza.rs y modelo.rs, y las de la semilla (usan la GPU)
./target/release/umap-mcqueen Waveforms.csv --train 2000 --lote 200 --semilla 7 --modelo modelos/m7
```

## La semilla

`--semilla <n>` (por defecto 9999) siembra **todos** los generadores del entrenamiento: la inicialización de la red en
el backend y el muestreo de aristas de cada época. Con los mismos datos, la misma configuración y la misma semilla
salen los mismos pesos, byte a byte; con otra semilla, otro modelo. El manifiesto del modelo la guarda (`"semilla"`,
formato 4); los manifiestos de formato 3, anteriores, se siguen cargando con la semilla desconocida.

Comprobarlo es `cargo test --release --test semilla`: entrena dos veces con la semilla 7 y una con la 8, carga los
tres modelos y proyecta los mismos datos con ellos; los dos de la semilla 7 coinciden coordenada a coordenada
(diferencia máxima 0) y el de la 8 no. Lo mismo, con `fast-umap` directo, para el encaje de entrenamiento.

Lo que **no** sirve es comparar los `.bin` byte a byte: burn sortea un identificador por parámetro al crear la red
y lo escribe en el fichero, así que dos modelos con los mismos pesos difieren en unos 50 bytes (medido: 51 de 13 022
con la misma semilla, frente a 12 190 con otra). Y lo que la semilla **no** fija es lo que viene después del encaje:
DenStream (`clump`) tiene su propio azar.

Antes de esto, dos entrenamientos idénticos diferían hasta un 19 % de la escala del encaje: `fast-umap` 1.6.0 solo
siembra el backend, y sus generadores de CPU salían de la entropía.

## fast-umap: por qué un `[patch]`, y cuándo se retira

`fast-umap` 1.6.0 (crates.io) no deja fijar la semilla. Mientras upstream no publique una versión que sí, `Cargo.toml`
sustituye el crate por el fork [`josemal98/fast-umap`](https://github.com/josemal98/fast-umap), rama `seed-config`:
un solo commit sobre el `master` de upstream (que es la 1.6.0 exacta) con `UmapConfig::seed`, propuesto a upstream
como PR. `Cargo.lock` fija el commit.

**Se retira** el día que crates.io traiga `fast-umap` con semilla configurable: se borra el bloque `[patch.crates-io]`
de `Cargo.toml`, se sube la versión de `fast-umap` y se archiva el fork. Nada más cambia.
