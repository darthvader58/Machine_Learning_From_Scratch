#!/usr/bin/env bash
# Two-pass ffmpeg GIF encoder.
#
# Pass 1 (palettegen) scans the WHOLE clip and builds one optimal 256-colour
# palette for it. Pass 2 (paletteuse) re-encodes against that palette with
# dithering. This beats a naive per-frame palette by 3-5x on size at equal or
# better quality, which is what keeps 720p Manim output inside the budget.
#
# usage: gif_from_mp4.sh <in.mp4> <out.gif> [width=640] [fps=15]
set -euo pipefail
IN="$1"; OUT="$2"; W="${3:-640}"; FPS="${4:-15}"
PAL="$(mktemp -t palette).png"
trap 'rm -f "$PAL"' EXIT

FILTERS="fps=${FPS},scale=${W}:-1:flags=lanczos"
ffmpeg -v error -y -i "$IN" -vf "${FILTERS},palettegen=stats_mode=diff" "$PAL"
ffmpeg -v error -y -i "$IN" -i "$PAL" \
  -lavfi "${FILTERS} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  "$OUT"

echo "$(basename "$OUT"): $(du -h "$OUT" | cut -f1)"
