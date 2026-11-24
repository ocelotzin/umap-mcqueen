#!/usr/bin/env sh
# Etiqueta con umap-mcqueen (EROLSC) cada spikes.csv de una carpeta y deja junto a
# cada uno erolsc_labels.csv (una etiqueta por forma de onda, −1 = anomalía),
# erolsc_encaje.csv (el encaje sobre el que agrupó) y erolsc.log. Mismo
# entrenamiento que mcqueen_labels.csv de la rama dev: UMAP con las primeras 2000
# formas de onda del propio archivo, semilla 9999, lotes de 200. Con la misma
# semilla el encaje es idéntico al de dev (encaje.csv), así que la comparación
# entre los dos agrupadores no depende del encaje.
#
#   sh scripts/etiqueta_erolsc.sh <carpeta> [paralelo=4] [semilla=9999] [sufijo=]
#
# Con un sufijo, los archivos se llaman erolsc_labels<sufijo>.csv, etc. (para
# guardar corridas con otras semillas sin pisar la principal).
set -eu
carpeta=${1:?uso: etiqueta_erolsc.sh <carpeta> [paralelo] [semilla] [sufijo]}
paralelo=${2:-4}; semilla=${3:-9999}; sufijo=${4:-}
train=2000; lote=200
raiz=$(cd "$(dirname "$0")/.." && pwd)
cargo build --release --manifest-path "$raiz/Cargo.toml" --bin umap-mcqueen
cli="$raiz/target/release/umap-mcqueen"
[ -x "$cli" ] || cli="$cli.exe"
find "$carpeta" -name spikes.csv | sort | xargs -P "$paralelo" -I{} sh -c '
  d=$(dirname "$1")
  if "$2" "$1" --train "$3" --lote "$4" --semilla "$5" --etiquetas "$d/erolsc_labels$6.csv" \
       --encaje "$d/erolsc_encaje$6.csv" > "$d/erolsc$6.log" 2>&1; then
    echo "$d: $(wc -l < "$d/erolsc_labels$6.csv") etiquetas (semilla $5)"
  else
    echo "$d: ERROR (ver $d/erolsc$6.log)"
  fi' _ {} "$cli" "$train" "$lote" "$semilla" "$sufijo"
