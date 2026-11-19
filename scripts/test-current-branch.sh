#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/test-current-branch.sh [all|unit|smoke]

  all    Run the unit suite and a small synthetic benchmark (default).
  unit   Run only unittest discovery.
  smoke  Run only the synthetic benchmark in a temporary directory.

Set PYTHON=/path/to/python to select an interpreter. Install the checked-out
branch first with: python -m pip install -e .
For Twitter branches, use: python -m pip install -e '.[twitter]'
EOF
}

mode="${1:-all}"
case "$mode" in
  all|unit|smoke) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$mode" == "all" || "$mode" == "unit" ]]; then
  echo "==> Running unit tests with $python_bin"
  "$python_bin" -m unittest discover -s "$repo_root/tests" -v
fi

if [[ "$mode" == "all" || "$mode" == "smoke" ]]; then
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/temporal-ann-smoke.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  echo "==> Running synthetic smoke benchmark in $temporary"
  "$python_bin" -m temporal_ann.benchmark \
    --size 2000 \
    --dimensions 16 \
    --queries 3 \
    --widths 0.01,0.1 \
    --fragments 1,4 \
    --ks 10 \
    --data-dir "$temporary/data" \
    --output "$temporary/results"
  test -s "$temporary/results/summary.json"
  echo "==> Smoke benchmark produced results/summary.json"
fi

echo "==> Branch test completed successfully"
