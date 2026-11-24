# Precalcula las referencias offline de cada canal de una carpeta: UMAP de R +
# HDBSCAN y PCA + HDBSCAN (umap_hdbscan_labels.csv, pca_hdbscan_labels.csv,
# umap_encaje.csv, pca_encaje.csv). Usa las MISMAS funciones que los Rmd: las
# toma del chunk `funciones` de analisis/comparacion_cuatro_etiquetados.Rmd.
#
# Los Rmd las calculan solos si faltan, pero canal por canal y con un hilo; con
# decenas de canales conviene repartirlas en varios procesos:
#
#   for k in 0 1 2 3 4 5; do Rscript scripts/referencias_offline.R data/mearec $k 6 & done; wait
#
# Argumentos: carpeta, parte (0..partes-1), partes. Correr desde la raíz del repo.
# Un canal que ya tiene sus cuatro archivos no se recalcula.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("uso: Rscript scripts/referencias_offline.R <carpeta> [parte] [partes]")
carpeta <- args[1]
parte <- if (length(args) >= 2) as.integer(args[2]) else 0L
partes <- if (length(args) >= 3) as.integer(args[3]) else 1L
suppressPackageStartupMessages({library(readr); library(dplyr); library(uwot); library(dbscan)})

lineas <- readLines("analisis/comparacion_cuatro_etiquetados.Rmd", encoding = "UTF-8")
ini <- which(startsWith(lineas, "```{r funciones"))
fin <- ini + which(lineas[(ini + 1):length(lineas)] == "```")[1]
eval(parse(text = lineas[(ini + 1):(fin - 1)]))

dirs <- sort(dirname(list.files(carpeta, pattern = "^spikes[.]csv$", recursive = TRUE, full.names = TRUE)))
dirs <- dirs[file.exists(file.path(dirs, "labels.csv"))]
dirs <- dirs[seq_along(dirs) %% partes == parte]
for (dir in dirs) {
  t0 <- Sys.time()
  X <- as.matrix(read_csv(file.path(dir, "spikes.csv"), col_names = FALSE, show_col_types = FALSE, progress = FALSE))
  # Mismos parámetros que los Rmd por defecto: min_pts automático, muestra de 5000, semilla 1.
  invisible(offline(dir, X, 0, 5000, 1))
  cat(basename(dir), nrow(X), "espigas,", round(as.numeric(Sys.time() - t0, units = "secs")), "s\n")
}
