#!/bin/bash
# build.sh — Compile custom radiation library and solver
# Run this after sourcing the OpenFOAM environment.
#
# Usage:
#   source /opt/openfoam2112/etc/bashrc
#   bash build.sh

set -e   # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Building IR Plume Analytics — Non-Gray fvDOM Solver"
echo "OpenFOAM version: $(foamVersion 2>/dev/null || echo 'unknown')"
echo "============================================================"

# ── Step 1: Compile the radiation library ─────────────────────────────────────
echo ""
echo "[1/2] Compiling libmyRadiation ..."
cd "$SCRIPT_DIR/src/myRadiation"
wmake libso
echo "      → $(ls $FOAM_USER_LIBBIN/libmyRadiation.so 2>/dev/null && echo OK || echo FAILED)"

# ── Step 2: Compile the solver application ────────────────────────────────────
echo ""
echo "[2/2] Compiling myRadiationSolver ..."
cd "$SCRIPT_DIR/src/myRadiationSolver"
wmake
echo "      → $(ls $FOAM_USER_APPBIN/myRadiationSolver 2>/dev/null && echo OK || echo FAILED)"

echo ""
echo "============================================================"
echo "Build complete."
echo ""
echo "Next steps:"
echo "  1. cd irSignature"
echo "  2. blockMesh"
echo "  3. checkMesh"
echo "  4. python ../scripts/map_plume_data.py --synthetic"
echo "  5. myRadiationSolver"
echo "  6. python ../scripts/extract_ir.py"
echo "============================================================"
