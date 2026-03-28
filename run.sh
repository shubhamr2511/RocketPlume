#!/bin/bash
# run.sh — Full pipeline runner
# Runs the complete IR signature estimation workflow.
#
# Usage:
#   source /opt/openfoam2112/etc/bashrc
#   bash run.sh [--parallel N] [--synthetic] [--skip-build]
#
# Options:
#   --parallel N    Run OpenFOAM solver on N cores (default: serial)
#   --synthetic     Use synthetic plume data (no real CFD data needed)
#   --skip-build    Skip compilation (if already built)
#   --skip-hitemp   Skip HITEMP processing (use existing kappa_lookup.dat)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$SCRIPT_DIR/irSignature"
N_PROCS=1
USE_SYNTHETIC=0
SKIP_BUILD=0
SKIP_HITEMP=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel) N_PROCS="$2"; shift 2 ;;
        --synthetic) USE_SYNTHETIC=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-hitemp) SKIP_HITEMP=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "IR Plume Analytics — Full Pipeline"
echo "Case: $CASE_DIR"
echo "Cores: $N_PROCS"
echo "============================================================"

# ── Phase 0: Build ────────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo ""
    echo "=== Phase 0: Build ==="
    bash "$SCRIPT_DIR/build.sh"
fi

# ── Phase 1: Mesh generation ──────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Mesh Generation ==="
cd "$CASE_DIR"

echo "  blockMesh ..."
blockMesh > log.blockMesh 2>&1
echo "  checkMesh ..."
checkMesh > log.checkMesh 2>&1
grep -E "(cells|faces|Max.*non|FAILED|No errors)" log.checkMesh | head -20

# ── Phase 3: Data mapping ─────────────────────────────────────────────────────
echo ""
echo "=== Phase 3: Data Mapping ==="

# HITEMP processing (skip if lookup table already exists or --skip-hitemp)
LOOKUP="$CASE_DIR/hitemp_data/kappa_lookup.dat"
if [[ $SKIP_HITEMP -eq 0 ]] && [[ ! -f "$LOOKUP" ]]; then
    echo "  Processing HITEMP data ..."
    cd "$SCRIPT_DIR"
    python3 scripts/process_hitemp.py \
        --species H2O CO2 CO \
        --data-dir "$CASE_DIR/hitemp_data" \
        --output "$LOOKUP"
fi

# Map plume data onto mesh
echo "  Mapping plume data ..."
cd "$SCRIPT_DIR"
if [[ $USE_SYNTHETIC -eq 1 ]]; then
    python3 scripts/map_plume_data.py \
        --synthetic \
        --case "$CASE_DIR" \
        --lookup "$LOOKUP"
else
    python3 scripts/map_plume_data.py \
        --case "$CASE_DIR" \
        --lookup "$LOOKUP"
fi

# ── Phase 4/5: Solve ──────────────────────────────────────────────────────────
echo ""
echo "=== Phase 5: Run Solver ==="
cd "$CASE_DIR"

if [[ $N_PROCS -gt 1 ]]; then
    echo "  Decomposing for $N_PROCS cores ..."
    decomposePar > log.decomposePar 2>&1
    echo "  Running myRadiationSolver (parallel) ..."
    mpirun -np "$N_PROCS" myRadiationSolver -parallel > log.solver 2>&1
    echo "  Reconstructing ..."
    reconstructPar > log.reconstructPar 2>&1
else
    echo "  Running myRadiationSolver (serial) ..."
    myRadiationSolver > log.solver 2>&1
fi

# Print convergence summary
echo "  Convergence summary:"
grep -E "(band|residual|converged|SIMPLE)" log.solver | tail -20

# ── Phase 6: Post-processing ──────────────────────────────────────────────────
echo ""
echo "=== Phase 6: Post-Processing ==="
cd "$SCRIPT_DIR"

python3 scripts/validate_gray.py --case "$CASE_DIR" --band 0 --ray 0

python3 scripts/extract_ir.py \
    --case "$CASE_DIR" \
    --time 1 \
    --sensor outlet

echo ""
echo "============================================================"
echo "Pipeline complete."
echo ""
echo "Results:"
echo "  ir_signature.txt  — per-band power summary"
echo "  ir_spectrum.png   — spectral distribution plot"
echo ""
echo "Visualise: paraFoam -case irSignature"
echo "============================================================"
