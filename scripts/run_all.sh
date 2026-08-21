#!/usr/bin/env bash
# Run the full pipeline: data -> validation -> backtests -> sensitivities -> report.
# Each stage gates the next; pass a stage name to resume from it:
#   ./scripts/run_all.sh [data|validate|backtests|sensitivities|report]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

stage="$1"
run() { echo -e "\n=== $1 ==="; shift; "$@"; }

case "$stage" in
  data)          run "Phase 1: download data"        "$PY" scripts/01_download_data.py ;&
  validate)      run "Phase 2: validate pricing"     "$PY" scripts/02_validate_pricing.py ;&
  backtests)     run "Phase 3: run backtests"        "$PY" scripts/03_run_backtests.py ;&
  sensitivities) run "Phase 4: sensitivity grid"     "$PY" scripts/04_sensitivities.py ;&
  report)        run "Phase 5: analysis + report"
                 "$PY" leaps_ls/analysis/decomposition.py
                 "$PY" leaps_ls/analysis/attribution.py
                 "$PY" leaps_ls/analysis/plots.py
                 "$PY" scripts/05_make_report.py ;;
  *) echo "usage: $0 [data|validate|backtests|sensitivities|report]"; exit 2 ;;
esac
echo -e "\nPipeline complete through stage: $stage"
