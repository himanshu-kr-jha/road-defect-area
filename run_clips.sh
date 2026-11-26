#!/usr/bin/env bash
# Run every clip listed in a clips file. Each line: <name> <video> <gpx> <height_m>
# Video and GPX can be local paths, http(s) URLs or s3:// URIs.
#
#   MODEL=/path/to/model.pt ./run_clips.sh clips/krrda_2026-11-26.txt
#
# Results go to out/<name>/. Open out/<name>/report.html and the out/<name>/defects/ images.
set -euo pipefail
cd "$(dirname "$0")"
list=${1:?usage: MODEL=/path/to/model.pt ./run_clips.sh clips/FILE.txt}
prod=/home/ubuntu/vision-stack/infrastructure/triton/models/post_cons/3/model.pt
MODEL=${MODEL:-$( [ -f "$prod" ] && echo "$prod" || echo model.pt )}
[ -f "$MODEL" ] || { echo "model not found: $MODEL (set MODEL=/path/to/model.pt)"; exit 1; }
grep -vE '^\s*(#|$)' "$list" | while read -r name video gpx height; do
  echo "=== $name"
  .venv/bin/python road_area.py run --video "$video" --gpx "$gpx" --height "$height" --model "$MODEL" --out "out/$name" ${EXTRA_ARGS:-} < /dev/null 2>&1 | tee "out/$name.log"
done
echo "done. Results: $(ls -d out/*/ | tr '\n' ' ')"
