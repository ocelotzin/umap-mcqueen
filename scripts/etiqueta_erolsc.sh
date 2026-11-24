#!/usr/bin/env sh
# Etiqueta con umap-mcqueen (EROLSC) cada spikes.csv de una carpeta y deja junto a
# cada uno erolsc_labels.csv (una etiqueta por forma de onda, −1 = anomalía) y
# erolsc.log. Mismo entrenamiento que mcqueen_labels.csv de la rama dev: UMAP con
# las primeras 2000 formas de onda del propio archivo, lotes de 200.
#
#   sh scripts/etiqueta_erolsc.sh <carpeta> [paralelo=4] [train=2000] [lote=200]
set -eu
carpeta=${1:?uso: etiqueta_erolsc.sh <carpeta> [paralelo] [train] [lote]}
paralelo=${2:-4}; train=${3:-2000}; lote=${4:-200}
raiz=$(cd "$(dirname "$0")/.." && pwd)
cargo build --release --manifest-path "$raiz/Cargo.toml" --bin umap-mcqueen
cli="$raiz/target/release/umap-mcqueen"
[ -x "$cli" ] || cli="$cli.exe"
find "$carpeta" -name spikes.csv | sort | xargs -P "$paralelo" -I{} sh -c '
  d=$(dirname "$1")
  if "$2" "$1" --train "$3" --lote "$4" --etiquetas "$d/erolsc_labels.csv" > "$d/erolsc.log" 2>&1; then
    echo "$d: $(wc -l < "$d/erolsc_labels.csv") etiquetas"
  else
    echo "$d: ERROR (ver $d/erolsc.log)"
  fi' _ {} "$cli" "$train" "$lote"
