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

## Datos simulados con MEArec

`scripts/simula_monocanal_mearec.py` simula registros extracelulares con [MEArec](https://mearec.readthedocs.io)
(neuronas biofísicas de capa 5 del Blue Brain Project alrededor de una sonda Neuronexus-32) y exporta canales
sueltos como electrodos monocanal, con los mismos archivos que `monocanal` (`spikes.csv`, `labels.csv`,
`times.csv`, `unidades.csv`). Cada registro sortea, con una semilla maestra, el número de neuronas, las tasas,
las ráfagas, el ruido, la deriva y las amplitudes dentro de rangos fisiológicos (`RANGOS` en el script).

Generar plantillas necesita NEURON 8 y LFPy, que no funcionan en Windows: se corre en WSL. Preparación (Arch,
como root; en otra distribución, el equivalente de `gcc make perl`):

```sh
pacman -S --needed gcc make perl                  # NEURON compila los mecanismos .mod con ellos
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, en ~/.local/bin
uv venv ~/mearec-env --python 3.12                # NEURON 8 no tiene versión para Python 3.14
uv pip install --python ~/mearec-env/bin/python 'MEArec[templates]'
```

Si una generación de plantillas falla a medias (por ejemplo, porque faltaba `perl`), MEArec deja la carpeta
`~/.config/mearec/<versión>/cell_models/bbp/mods` sin compilar y no la vuelve a compilar: hay que borrarla.

```sh
PY=~/mearec-env/bin/python
# Una vez: 780 plantillas con deriva a 30 kHz (unos 7 min con 20 núcleos, ~670 MB)
$PY scripts/simula_monocanal_mearec.py plantillas --out ~/mearec/plantillas_nnx32_30k.h5
# Registros: 8 canales al azar por registro, en serie para que la misma semilla dé exactamente
# los mismos datos (~4 min por registro de 1200 s). Desde la raíz del repo, en WSL:
$PY scripts/simula_monocanal_mearec.py registros --plantillas ~/mearec/plantillas_nnx32_30k.h5 \
    --out data/mearec --n-registros 6 --duracion 1200 --semilla 1
```

La frecuencia de muestreo se fija con las plantillas (`--fs-khz`, 30 por defecto): MEArec 1.11 falla al
remuestrear plantillas con deriva, así que los registros salen siempre a la frecuencia de su biblioteca.

-----

## Reproducibilidad de los reportes

Los informes comparan EROLSC (esta rama) con la propagación de etiquetas de la rama `dev` y con dos referencias
offline (UMAP de R + HDBSCAN, PCA + HDBSCAN), contra la verdad de campo, en tres conjuntos de datos:

| Conjunto | Carpeta | Papel |
|---|---|---|
| Registro híbrido (Janelia, 16 canales, 1200 s) | `data/monocanal` | fuera de muestra |
| Simulados con MEArec, semilla maestra 1 (6 registros) | `data/mearec` | **calibración** de EROLSC |
| Simulados con MEArec, semilla maestra 2 (6 registros) | `data/mearec_validacion` | **validación** |

Los parámetros por defecto de EROLSC (ventana 2000, `Td` 4.5, `Tn` 0.3, `M` 100; ver `src/erolsc.rs`) se eligieron
sólo con `data/mearec`. `data/` y `reportes/` no se versionan: todo lo que contienen se regenera con lo que sigue.

**Organización.** `analisis/` tiene los dos Rmd (fuente); `reportes/` recibe los HTML; `data/<conjunto>/` tiene una
subcarpeta por canal, con las formas de onda (`spikes.csv`), la verdad (`labels.csv`) y lo que se genera junto a ellas.

**Versiones con que se generaron.** Rust 1.99 (fast-umap: el fork con semilla, fijado en `Cargo.toml`); rama `dev` en
el commit `7cc6874`; R 4.6.2 con readr, dplyr, tidyr, ggplot2, uwot, dbscan y rmarkdown; pandoc de RStudio; en WSL
(Arch), Python 3.12.16 con MEArec 1.11.0, NEURON 8.2.7 y LFPy 2.3.6.

### 1. Datos

- `data/monocanal`: canales del registro híbrido `sub-hybrid-janelia_ses-hybrid-drift-siprobe-rec-16c-1200s-31_ecephys.nwb`,
  exportados con `scripts/hibrido_a_monocanal.py` de la rama `dev`.
- `data/mearec` y `data/mearec_validacion`: la biblioteca de plantillas (semilla 0) y los registros de la sección
  anterior, con `--semilla 1` y `--semilla 2` respectivamente (6 registros de 1200 s cada uno). Para ir más rápido se
  pueden simular por tandas en varios procesos con `--primero 0/2/4 --n-registros 2`: cada registro depende sólo de la
  semilla maestra y de su índice. Los canales con menos de 2000 espigas no se pueden etiquetar (el UMAP se entrena con
  2000) y quedan fuera de la comparación.

### 2. Etiquetas (UMAP con semilla 9999, entrenado con las primeras 2000 espigas, lotes de 200)

```sh
cargo build --release && cargo build --release --example sonda_de_erolsc
for c in data/monocanal data/mearec data/mearec_validacion; do
  sh scripts/etiqueta_erolsc.sh $c 3        # erolsc_labels.csv y erolsc_encaje.csv
  sh scripts/etiqueta_dev.sh   $c 3         # mcqueen_labels.csv (dev en 7cc6874, worktree temporal)
done
# Otras semillas de UMAP en monocanal (sección «Otras semillas» del informe):
for s in 1 2 3; do sh scripts/etiqueta_erolsc.sh data/monocanal 3 $s _s$s; sh scripts/etiqueta_dev.sh data/monocanal 3 $s _s$s; done
```

Con la misma semilla, el encaje de esta rama es idéntico, valor por valor, al de `dev`: la comparación sólo depende
del agrupador.

### 3. Rejillas de EROLSC

Calibración, sobre `data/mearec` (archivos `erolsc_cal_*.csv`), y la elección de parámetros:

```sh
for d in data/mearec/r*/r*_canal_*; do
  [ -f $d/erolsc_encaje.csv ] || continue
  SONDA_TD=1.5,2.0,2.5,3.0,3.5,4.0 SONDA_TN=0.1,0.3,0.5,0.7 SONDA_M=20,50,100,200 \
    ./target/release/examples/sonda_de_erolsc $d/erolsc_encaje.csv $d/labels.csv 200 500,1000,2000,4000 > $d/erolsc_cal_v.csv
  SONDA_TD=3.0,4.0,4.5,5.0,6.0 SONDA_TN=0.3,0.5,0.7 SONDA_M=20,50,100 \
    ./target/release/examples/sonda_de_erolsc $d/erolsc_encaje.csv $d/labels.csv 200 1500,2000,3000 > $d/erolsc_cal_ext.csv
done
python scripts/analiza_calibracion.py data/mearec
```

Validación cruzada y barrido de ventana en `data/monocanal` (secciones del informe; no intervienen en la calibración):

```sh
for d in data/monocanal/canal_*; do
  ./target/release/examples/sonda_de_erolsc $d/erolsc_encaje.csv $d/labels.csv > $d/erolsc_rejilla.csv
  ./target/release/examples/sonda_de_erolsc $d/erolsc_encaje.csv $d/labels.csv 200 500,1000,2000,4000,8000 > $d/erolsc_rejilla_ventana.csv
done
```

### 4. Referencias offline

Los Rmd calculan UMAP + HDBSCAN y PCA + HDBSCAN de cada canal si faltan, pero con un hilo; con muchos canales
conviene precalcularlas en paralelo (mismas funciones y parámetros que los Rmd):

```sh
for c in data/monocanal data/mearec data/mearec_validacion; do
  for k in 0 1 2 3 4 5; do Rscript scripts/referencias_offline.R $c $k 6 & done; wait
done
```

### 5. Informes

Desde la raíz del repo (en Windows, fuera de RStudio, antes `export RSTUDIO_PANDOC="C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools"`).
`carpeta` es relativa a `analisis/`; `fuera_de_muestra = FALSE` sólo para los datos de calibración:

```sh
for rmd in comparacion_cuatro_etiquetados comparacion_erolsc_dev_umap_hdbscan; do
  Rscript -e "rmarkdown::render('analisis/$rmd.Rmd', output_dir = 'reportes', output_file = '${rmd}_monocanal.html')"
  Rscript -e "rmarkdown::render('analisis/$rmd.Rmd', output_dir = 'reportes', output_file = '${rmd}_mearec_calibracion.html',
              params = list(carpeta = '../data/mearec', fuera_de_muestra = FALSE))"
  Rscript -e "rmarkdown::render('analisis/$rmd.Rmd', output_dir = 'reportes', output_file = '${rmd}_mearec_validacion.html',
              params = list(carpeta = '../data/mearec_validacion'))"
done
```

Cada informe lleva en el nombre el conjunto de datos: `_monocanal` (registro híbrido), `_mearec_calibracion`
(`data/mearec`) y `_mearec_validacion` (`data/mearec_validacion`).

Resultado con los parámetros actuales (ARI medio contra la verdad; EROLSC / propagación de dev / UMAP + HDBSCAN):
validación 0.428 / 0.407 / 0.573; registro híbrido 0.173 / 0.171 / 0.172. EROLSC y la propagación quedan empatados
(intervalos de confianza que incluyen el cero); EROLSC recupera algunas neuronas completas más.
