#!/usr/bin/env bash

set -e

echo "running dolfinx"
apptainer exec dolfinx.sif python3 solve_dolfinx_2.py

echo "done. plotting"
uv run plot_local.py

echo "done."