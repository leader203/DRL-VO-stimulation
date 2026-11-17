#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /usr/bin/python3 -s /home/user/workspace/social_navigation_runtime_20261010/experiments/newmetro_pedestrians_final_demo_20261020T214339/run_demo.py "$@"
