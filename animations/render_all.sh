#!/usr/bin/env bash
# Rebuild every animation GIF from source.
#
# The GIFs are build artifacts; these scripts are the source of truth. Every
# script seeds its RNG, so a rebuild reproduces the committed GIFs byte for
# byte - the same property that let this repo be regenerated after its
# notebooks were lost.
#
# Run from the repo root:  bash animations/render_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/opt/anaconda3/envs/tf_mps/bin/python

echo "=== matplotlib scenes ==="
for f in animations/mpl_scenes/*.py; do
  echo "--- $(basename "$f")"
  "$PY" "$f"
done

if [ -d animations/manim_scenes ] && compgen -G "animations/manim_scenes/*.py" > /dev/null; then
  echo "=== manim scenes ==="
  echo "(these need BasicTeX + the tlmgr packages listed in animations/README.md)"
  for f in animations/manim_scenes/*.py; do
    echo "--- $(basename "$f")"
    "$PY" "$f"
  done
fi

echo
echo "=== sizes ==="
ls -la animations/gifs/*.gif | awk '{printf "  %-34s %6.2f MB\n", $9, $5/1e6}'
