#!/usr/bin/env python3
"""
map_plume_data.py
-----------------
Map tabular plume CFD data and HITEMP kappa lookup table onto an OpenFOAM mesh.

Workflow:
  1. Read source data: CSV with columns (x, r, T, p, XH2O, XCO2, XCO)
  2. Read the OpenFOAM mesh cell centres from constant/polyMesh/
  3. Interpolate source data at each cell centre (x, r)
  4. Write volScalarField files: T, p, XH2O, XCO2, XCO
  5. Read kappa_lookup.dat and map kappaBand0/1/2 for each cell

Usage:
    python scripts/map_plume_data.py \
        --source plume_data.csv \
        --case irSignature \
        --lookup irSignature/hitemp_data/kappa_lookup.dat

Source CSV format (header required):
    x[m], r[m], T[K], p[Pa], XH2O[-], XCO2[-], XCO[-]

The script also accepts OpenFOAM mapFields output — if the source case is
already an OpenFOAM case, use mapFields directly and skip this script.
"""

import argparse
import os
import sys
import struct
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import warnings
warnings.filterwarnings("ignore")


# ── OpenFOAM volScalarField writer ───────────────────────────────────────────

OF_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Version:  v2112
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      {name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      {dims};

internalField   nonuniform List<scalar>
{n_cells}
(
"""

OF_FOOTER = """\
);

boundaryField
{{
{boundaries}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

# Dimension sets for each field
DIMENSIONS = {
    "T":          "[0 0 0 1 0 0 0]",
    "p":          "[1 -1 -2 0 0 0 0]",
    "XH2O":       "[0 0 0 0 0 0 0]",
    "XCO2":       "[0 0 0 0 0 0 0]",
    "XCO":        "[0 0 0 0 0 0 0]",
    "kappaBand0": "[0 -1 0 0 0 0 0]",
    "kappaBand1": "[0 -1 0 0 0 0 0]",
    "kappaBand2": "[0 -1 0 0 0 0 0]",
}

# Boundary conditions for each field (patch name: type + value)
BOUNDARY_TEMPLATE = {
    "T": {
        "inlet":      "fixedValue; value uniform 900;",
        "outlet":     "zeroGradient;",
        "outerWall":  "fixedValue; value uniform 288;",
        "wedgeFront": "wedge;",
        "wedgeBack":  "wedge;",
        "axis":       "empty;",
    },
    "p": {
        "inlet":      "fixedValue; value uniform 101325;",
        "outlet":     "zeroGradient;",
        "outerWall":  "zeroGradient;",
        "wedgeFront": "wedge;",
        "wedgeBack":  "wedge;",
        "axis":       "empty;",
    },
    "XH2O": {
        "inlet":      "fixedValue; value uniform 0.12;",
        "outlet":     "zeroGradient;",
        "outerWall":  "fixedValue; value uniform 0.015;",
        "wedgeFront": "wedge;",
        "wedgeBack":  "wedge;",
        "axis":       "empty;",
    },
    "XCO2": {
        "inlet":      "fixedValue; value uniform 0.08;",
        "outlet":     "zeroGradient;",
        "outerWall":  "fixedValue; value uniform 0.0004;",
        "wedgeFront": "wedge;",
        "wedgeBack":  "wedge;",
        "axis":       "empty;",
    },
    "XCO": {
        "inlet":      "fixedValue; value uniform 0.005;",
        "outlet":     "zeroGradient;",
        "outerWall":  "fixedValue; value uniform 0.0;",
        "wedgeFront": "wedge;",
        "wedgeBack":  "wedge;",
        "axis":       "empty;",
    },
}
# kappaBand fields get same BC as each other
for bn in ["kappaBand0", "kappaBand1", "kappaBand2"]:
    BOUNDARY_TEMPLATE[bn] = {
        "inlet":      "zeroGradient;",
        "outlet":     "zeroGradient;",
        "outerWall":  "fixedValue; value uniform 1e-10;",
        "wedgeFront": "wedge;",
        "wedgeBack":  "wedge;",
        "axis":       "empty;",
    }


def write_volscalarfield(case_dir: Path, name: str, values: np.ndarray,
                          boundary_patches: dict | None = None):
    """Write an OpenFOAM volScalarField ASCII file."""
    out_path = case_dir / "0" / name
    n_cells  = len(values)
    dims     = DIMENSIONS[name]

    bc = boundary_patches or BOUNDARY_TEMPLATE.get(name, {})
    bc_str = ""
    for patch, bc_def in bc.items():
        bc_str += f"    {patch}\n    {{\n        type            {bc_def}\n    }}\n\n"

    with open(out_path, "w") as f:
        f.write(OF_HEADER.format(name=name, dims=dims, n_cells=n_cells))
        for v in values:
            f.write(f"{v:.8e}\n")
        f.write(OF_FOOTER.format(boundaries=bc_str))

    print(f"  Wrote {out_path}  (min={values.min():.4g}, max={values.max():.4g})")


# ── OpenFOAM mesh cell centre reader ─────────────────────────────────────────

def read_of_points(case_dir: Path) -> np.ndarray:
    """
    Read cell centres from OpenFOAM constant/polyMesh/C or
    from the C file written after 'postProcess -func writeCellCentres'.
    Falls back to reading from points file if C is not available.
    Returns array of shape (nCells, 3).
    """
    # Preferred: read from 0/C written by writeCellCentres
    c_file = case_dir / "0" / "C"
    if not c_file.exists():
        # Try postProcess to generate it
        print("  Cell centres file 0/C not found.")
        print("  Run: postProcess -func writeCellCentres -case <case_dir>")
        print("  Or provide source data on a regular (x,r) grid.")
        return None

    centres = []
    reading = False
    with open(c_file) as f:
        for line in f:
            line = line.strip()
            if line == "(":
                reading = True
                continue
            if reading:
                if line == ");":
                    break
                # Format: (x y z)
                xyz = line.strip("()").split()
                if len(xyz) == 3:
                    centres.append([float(v) for v in xyz])

    return np.array(centres)


# ── HITEMP lookup table reader and interpolator ───────────────────────────────

def load_kappa_lookup(lookup_path: Path) -> dict:
    """
    Read kappa_lookup.dat into structured numpy arrays.
    Returns dict with keys: T, p, X, kappa (shape: [nT, nP, nX, nBands])
    """
    print(f"  Loading kappa lookup table: {lookup_path}")
    data = np.loadtxt(lookup_path, comments="#")
    # Columns: T, p, X, kappa0, kappa1, kappa2
    T_vals = np.unique(data[:, 0])
    P_vals = np.unique(data[:, 1])
    X_vals = np.unique(data[:, 2])

    nT, nP, nX = len(T_vals), len(P_vals), len(X_vals)
    n_bands = data.shape[1] - 3

    kappa = np.zeros((nT, nP, nX, n_bands))
    for row in data:
        iT = np.searchsorted(T_vals, row[0])
        iP = np.searchsorted(P_vals, row[1])
        iX = np.searchsorted(X_vals, row[2])
        kappa[iT, iP, iX, :] = row[3:]

    print(f"    Grid: {nT} T × {nP} p × {nX} X, {n_bands} bands")
    return {
        "T": T_vals, "p": P_vals, "X": X_vals, "kappa": kappa,
        "n_bands": n_bands
    }


def interpolate_kappa(lookup: dict, T_cells: np.ndarray, p_cells: np.ndarray,
                       X_cells: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation of kappa lookup table at cell values.
    Returns array of shape (nCells, nBands).
    Clamps T, p, X to lookup table bounds.
    """
    T_g  = lookup["T"]
    P_g  = lookup["p"]
    X_g  = lookup["X"]
    k_3d = lookup["kappa"]      # (nT, nP, nX, nBands)
    nb   = lookup["n_bands"]

    # Clamp to grid bounds
    T_q = np.clip(T_cells, T_g.min(), T_g.max())
    P_q = np.clip(p_cells / 101325.0, P_g.min(), P_g.max())  # Pa → atm
    X_q = np.clip(X_cells, X_g.min(), X_g.max())

    result = np.zeros((len(T_cells), nb))
    for b in range(nb):
        interp = RegularGridInterpolator(
            (T_g, P_g, X_g), k_3d[:, :, :, b],
            method="linear", bounds_error=False, fill_value=1e-10
        )
        result[:, b] = interp(np.column_stack([T_q, P_q, X_q]))

    return np.maximum(result, 1e-10)


# ── Source data reader ────────────────────────────────────────────────────────

def load_source_csv(csv_path: Path) -> dict:
    """
    Read plume source data CSV.
    Expected columns: x, r, T, p, XH2O, XCO2, XCO
    """
    print(f"  Loading source data: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {
        "x":    data["x"],
        "r":    data["r"],
        "T":    data["T"],
        "p":    data["p"],
        "XH2O": data["XH2O"],
        "XCO2": data["XCO2"],
        "XCO":  data["XCO"],
    }


def interpolate_to_cells(src: dict, cell_centres: np.ndarray) -> dict:
    """
    Interpolate source fields from (x, r) grid to cell centres.
    Cells use (x, sqrt(y^2+z^2)) as radial coordinate.
    """
    x_src = src["x"]
    r_src = src["r"]

    # Cell radial distance
    cell_x = cell_centres[:, 0]
    cell_r = np.sqrt(cell_centres[:, 1]**2 + cell_centres[:, 2]**2)

    pts = np.column_stack([x_src, r_src])

    result = {}
    for field in ["T", "p", "XH2O", "XCO2", "XCO"]:
        interp = LinearNDInterpolator(pts, src[field], fill_value=np.nan)
        vals   = interp(np.column_stack([cell_x, cell_r]))

        # Fill NaN (outside convex hull) with boundary values
        if field == "T":
            fill = 288.0
        elif field == "p":
            fill = 101325.0
        else:
            fill = 0.0

        vals = np.where(np.isnan(vals), fill, vals)
        result[field] = vals

    return result


# ── Synthetic plume data (for testing without real CFD data) ──────────────────

def generate_synthetic_plume(n_x: int = 50, n_r: int = 30) -> dict:
    """
    Generate a synthetic Gaussian plume field for testing.
    Temperature: Gaussian in r, linearly decaying in x.
    Species: Gaussian profile following temperature.
    """
    print("  Generating synthetic Gaussian plume data for testing")

    x = np.linspace(0, 0.4, n_x)
    r = np.linspace(0, 0.015, n_r)
    xx, rr = np.meshgrid(x, r, indexing="ij")

    # Plume half-width grows linearly (spreading angle ~5°)
    r_half = 0.003 + 0.02 * xx    # half-width [m]

    # Temperature: peak 1200K at nozzle, decays axially and radially
    T_peak  = 1200.0 - 1800.0 * (xx / 0.4)
    T_peak  = np.maximum(T_peak, 288.0)
    T_gauss = T_peak * np.exp(-0.5 * (rr / r_half)**2)
    T_field = np.maximum(T_gauss, 288.0)

    # Species: proportional to temperature excess
    theta = (T_field - 288.0) / (T_peak - 288.0 + 1e-10)
    XH2O  = 0.12 * theta + 0.015 * (1.0 - theta)
    XCO2  = 0.08 * theta + 0.0004 * (1.0 - theta)
    XCO   = 0.005 * theta
    p_field = np.full_like(T_field, 101325.0)

    return {
        "x":    xx.ravel(), "r": rr.ravel(),
        "T":    T_field.ravel(), "p": p_field.ravel(),
        "XH2O": XH2O.ravel(), "XCO2": XCO2.ravel(), "XCO": XCO.ravel(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="Path to plume source CSV. If omitted, uses synthetic data.")
    parser.add_argument("--case", default="irSignature",
                        help="Path to OpenFOAM case directory")
    parser.add_argument("--lookup", default="irSignature/hitemp_data/kappa_lookup.dat",
                        help="Path to kappa lookup table")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic Gaussian plume (for testing)")
    args = parser.parse_args()

    case_dir    = Path(args.case)
    lookup_path = Path(args.lookup)

    print("=" * 60)
    print("Plume Data Mapper → OpenFOAM Fields")
    print("=" * 60)

    # ── Step 1: Load source data
    if args.synthetic or args.source is None:
        src = generate_synthetic_plume()
    else:
        src = load_source_csv(Path(args.source))

    # ── Step 2: Load cell centres
    print("\nReading cell centres from OpenFOAM mesh...")
    cell_centres = read_of_points(case_dir)

    if cell_centres is None:
        print("  Falling back to synthetic grid output (no mesh found)")
        # Write fields for a dummy 1-cell case for testing
        n = 100
        cell_centres = np.zeros((n, 3))
        cell_centres[:, 0] = np.linspace(0.001, 0.399, n)

    print(f"  {len(cell_centres)} cell centres loaded")

    # ── Step 3: Interpolate flow fields
    print("\nInterpolating flow fields onto mesh...")
    mapped = interpolate_to_cells(src, cell_centres)

    # ── Step 4: Write T, p, species fields
    print("\nWriting OpenFOAM field files...")
    for name in ["T", "p", "XH2O", "XCO2", "XCO"]:
        write_volscalarfield(case_dir, name, mapped[name])

    # ── Step 5: Map kappa from lookup table
    if lookup_path.exists():
        print("\nInterpolating kappa from HITEMP lookup table...")
        lookup = load_kappa_lookup(lookup_path)

        # Use total mole fraction (H2O dominant in MWIR band)
        X_total = mapped["XH2O"] + mapped["XCO2"] + mapped["XCO"]

        kappa_cells = interpolate_kappa(
            lookup, mapped["T"], mapped["p"], mapped["XH2O"]
        )

        for b in range(lookup["n_bands"]):
            write_volscalarfield(case_dir, f"kappaBand{b}", kappa_cells[:, b])
    else:
        print(f"\nWARNING: kappa lookup table not found at {lookup_path}")
        print("  Run process_hitemp.py first, or download HITEMP data.")
        print("  Writing placeholder kappa fields (1e-5 m^-1)...")
        for b in range(3):
            write_volscalarfield(
                case_dir, f"kappaBand{b}",
                np.full(len(cell_centres), 1e-5)
            )

    print("\nDone. Verify fields in ParaView: paraFoam -case", args.case)


if __name__ == "__main__":
    main()
