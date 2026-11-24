#!/usr/bin/env sh
# Etiqueta cada spikes.csv de una carpeta con el agrupador de la rama dev
# (propagación de etiquetas por vecindad), para compararlo con EROLSC. Deja junto
# a cada uno mcqueen_labels<sufijo>.csv y mcqueen<sufijo>.log.
#
# Compila dev en un worktree temporal, en el commit con el que se hicieron las
# comparaciones (7cc6874: reproduce exactamente los mcqueen_labels.csv de
# monocanal), con su propio CARGO_TARGET_DIR para no pisar el binario de esta
# rama. Mismo entrenamiento que scripts/etiqueta_erolsc.sh: UMAP con las primeras
# 2000 formas de onda, lotes de 200; con la misma semilla, el encaje es idéntico.
#
#   sh scripts/etiqueta_dev.sh <carpeta> [paralelo=3] [semilla=9999] [sufijo=] [commit=7cc6874]
set -eu
carpeta=$(cd "${1:?uso: etiqueta_dev.sh <carpeta> [paralelo] [semilla] [sufijo] [commit]}" && pwd)
paralelo=${2:-3}; semilla=${3:-9999}; sufijo=${4:-}; commit=${5:-7cc6874}
raiz=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
destino="${CARGO_TARGET_DIR_DEV:-$raiz/target/dev}"
trap 'git -C "$raiz" worktree remove --force "$tmp/dev" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
git -C "$raiz" worktree add --detach "$tmp/dev" "$commit" >/dev/null
(cd "$tmp/dev" && CARGO_TARGET_DIR="$destino" cargo build --release --bin umap-mcqueen)
cli="$destino/release/umap-mcqueen"
[ -x "$cli" ] || cli="$cli.exe"
# dev busca rutas relativas a su propia raíz: se corre desde el worktree, con rutas absolutas.
find "$carpeta" -name spikes.csv | sort | xargs -P "$paralelo" -I{} sh -c '
  d=$(dirname "$1")
  if (cd "$2" && "$3" "$1" --train 2000 --lote 200 --semilla "$4" --etiquetas "$d/mcqueen_labels$5.csv" \
        > "$d/mcqueen$5.log" 2>&1); then
    echo "$d: $(wc -l < "$d/mcqueen_labels$5.csv") etiquetas (dev, semilla $4)"
  else
    echo "$d: ERROR ($(grep -m1 -i error "$d/mcqueen$5.log"))"
  fi' _ {} "$tmp/dev" "$cli" "$semilla" "$sufijo"
