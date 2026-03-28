#!/usr/bin/env python3
"""
process_hitemp.py
-----------------
Parse HITEMP line-by-line .par files and generate a band-averaged absorption
coefficient lookup table for H2O, CO2, and CO.

Output: hitemp_data/kappa_lookup.dat
    Columns: T[K]  p[atm]  X_species[-]  kappa_band0[1/m]  kappa_band1[1/m]  kappa_band2[1/m]

Bands (wavenumbers in cm^-1):
    Band 0: 2000 – 3333  (MWIR 3–5 µm)
    Band 1: 1000 – 2000  (MIR 5–10 µm)
    Band 2:  714 – 1250  (LWIR 8–14 µm)

Usage:
    python scripts/process_hitemp.py --species H2O CO2 CO \
        --data-dir irSignature/hitemp_data \
        --output irSignature/hitemp_data/kappa_lookup.dat

References:
    HITRAN/HITEMP 160-char .par format: https://hitran.org/docs/definitions-and-units/
    Gordon et al., JQSRT 2017 (HITRAN 2016 format documentation)
    Chang & Rhee (1984) for fractional blackbody
"""

import argparse
import os
import sys
import numpy as np
from scipy.special import voigt_profile   # scipy >= 1.7 has this
from pathlib import Path
import itertools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Physical constants ────────────────────────────────────────────────────────
C1  = 1.19104e-8      # 2hc^2 [W·cm^2·sr^-1]  (first radiation constant)
C2  = 1.43877         # hc/k  [cm·K]           (second radiation constant)
KB  = 1.38065e-23     # Boltzmann constant [J/K]
NA  = 6.02214e23      # Avogadro number [mol^-1]
P0  = 1.0             # reference pressure [atm]
T0  = 296.0           # HITEMP reference temperature [K]

# ── IR band definitions ───────────────────────────────────────────────────────
BANDS = [
    (2000.0, 3333.0, "MWIR 3-5um"),
    (1000.0, 2000.0, "MIR 5-10um"),
    ( 714.0, 1250.0, "LWIR 8-14um"),
]
N_BANDS = len(BANDS)

# ── HITEMP 160-char format field slices ──────────────────────────────────────
# Reference: https://hitran.org/docs/definitions-and-units/
FIELD_SLICES = {
    "mol_id":       slice(0,   2),
    "iso":          slice(2,   3),
    "nu":           slice(3,  15),   # vacuum wavenumber [cm^-1]
    "S":            slice(15, 25),   # line intensity at 296K [cm^-1/(mol·cm^-2)]
    "A":            slice(25, 35),   # Einstein A coefficient (unused)
    "gamma_air":    slice(35, 40),   # air-broadened HWHM [cm^-1/atm]
    "gamma_self":   slice(40, 45),   # self-broadened HWHM [cm^-1/atm]
    "E_lower":      slice(45, 55),   # lower-state energy [cm^-1]
    "n_air":        slice(55, 59),   # temperature exponent for air broadening
    "delta_air":    slice(59, 67),   # pressure shift [cm^-1/atm]
}

# Molecule IDs in HITRAN database
MOL_IDS = {"H2O": 1, "CO2": 2, "CO": 5}

# Molecular masses [g/mol]
MOL_MASS = {"H2O": 18.015, "CO2": 44.010, "CO": 28.010}


# ── Lookup table grid ─────────────────────────────────────────────────────────
T_GRID   = np.arange(300, 3001, 100, dtype=float)    # 300–3000 K, 28 points
P_GRID   = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])  # atm
X_GRID   = np.array([0.001, 0.005, 0.01, 0.03, 0.05,
                     0.08, 0.10, 0.15, 0.20, 0.30])          # mole fraction


def parse_par_line(line: str) -> dict | None:
    """Parse one 160-character HITRAN .par line. Returns None if malformed."""
    if len(line) < 100:
        return None
    try:
        return {
            "mol_id":     int(line[FIELD_SLICES["mol_id"]]),
            "iso":        int(line[FIELD_SLICES["iso"]]),
            "nu":         float(line[FIELD_SLICES["nu"]]),
            "S":          float(line[FIELD_SLICES["S"]]),
            "gamma_air":  float(line[FIELD_SLICES["gamma_air"]]),
            "gamma_self": float(line[FIELD_SLICES["gamma_self"]]),
            "E_lower":    float(line[FIELD_SLICES["E_lower"]]),
            "n_air":      float(line[FIELD_SLICES["n_air"]]),
        }
    except (ValueError, IndexError):
        return None


def load_par_in_bands(par_path: str, mol_id: int) -> dict[int, dict]:
    """
    Parse .par file and return lines grouped by band index.
    Only loads wavenumbers that fall within any of the BANDS.
    Greatly reduces memory usage for large (multi-GB) H2O files.

    Returns: {band_idx: {"nu": array, "S": array, ...}}
    """
    nu_min_global = min(b[0] for b in BANDS)
    nu_max_global = max(b[1] for b in BANDS)

    # Pre-allocate per-band lists
    data = {b: {k: [] for k in ["nu", "S", "gamma_air", "gamma_self", "E_lower", "n_air"]}
            for b in range(N_BANDS)}

    par_path = Path(par_path)
    if not par_path.exists():
        print(f"  WARNING: {par_path} not found — skipping species")
        return data

    print(f"  Parsing {par_path.name} ...")
    n_total = 0
    n_loaded = 0

    with open(par_path, "r") as f:
        for line in f:
            n_total += 1
            rec = parse_par_line(line)
            if rec is None:
                continue
            if rec["mol_id"] != mol_id:
                continue
            nu = rec["nu"]
            if nu < nu_min_global or nu > nu_max_global:
                continue

            # Assign to correct band
            for bi, (nu_lo, nu_hi, _) in enumerate(BANDS):
                if nu_lo <= nu <= nu_hi:
                    data[bi]["nu"].append(rec["nu"])
                    data[bi]["S"].append(rec["S"])
                    data[bi]["gamma_air"].append(rec["gamma_air"])
                    data[bi]["gamma_self"].append(rec["gamma_self"])
                    data[bi]["E_lower"].append(rec["E_lower"])
                    data[bi]["n_air"].append(rec["n_air"])
                    n_loaded += 1
                    break

    # Convert to numpy arrays
    for bi in range(N_BANDS):
        for k in data[bi]:
            data[bi][k] = np.array(data[bi][k], dtype=float)

    total_band_lines = sum(len(data[bi]["nu"]) for bi in range(N_BANDS))
    print(f"    {n_total:,} lines scanned, {n_loaded:,} in bands of interest")
    return data


def partition_function_ratio(T: float, mol: str) -> float:
    """
    Q(T0) / Q(T) ratio using a simple polynomial fit.
    For a linear molecule (CO2, CO): Q ~ T (approximately)
    For non-linear (H2O): Q ~ T^1.5 (approximately)
    More accurate TIPS-2021 polynomials can replace this.
    """
    if mol == "H2O":
        # Non-linear triatomic: Q ∝ T^1.5
        return (T0 / T) ** 1.5
    else:
        # Linear diatomic/triatomic: Q ∝ T
        return T0 / T


def scale_line_intensity(S_ref: np.ndarray, E_lower: np.ndarray,
                          nu: np.ndarray, T: float, mol: str) -> np.ndarray:
    """
    Scale line intensity from reference temperature T0=296K to temperature T.
    HITRAN intensity scaling formula:

        S(T) = S(T0) * [Q(T0)/Q(T)]
                      * exp(-c2*E''/T) / exp(-c2*E''/T0)
                      * [1 - exp(-c2*nu/T)] / [1 - exp(-c2*nu/T0)]
    """
    Q_ratio   = partition_function_ratio(T, mol)
    boltzmann = np.exp(-C2 * E_lower / T) / np.exp(-C2 * E_lower / T0)
    stim_emis = (1.0 - np.exp(-C2 * nu / T)) / (1.0 - np.exp(-C2 * nu / T0))

    return S_ref * Q_ratio * boltzmann * stim_emis


def voigt_hwhm(gamma_L: float, gamma_D: float) -> float:
    """
    Pseudo-Voigt HWHM approximation (Thompson et al. 1987).
    f_V ≈ 0.5346*f_L + sqrt(0.2166*f_L^2 + f_D^2)
    """
    return 0.5346 * gamma_L + np.sqrt(0.2166 * gamma_L**2 + gamma_D**2)


def compute_band_kappa(band_data: dict, T: float, p: float, X: float,
                        mol: str, mol_mass: float) -> float:
    """
    Compute band-averaged absorption coefficient [1/m] for one (T, p, X) point.

    Method: sum cross-sections of all lines in the band, weighted by Voigt profile
    integrated over the band width (effectively the peak cross-section approximation
    since lines are narrow compared to band width).

    kappa [1/m] = n_mol [molecules/m^3] * sum_i(sigma_i) [m^2/molecule]

    where sigma_i = S_i(T) [cm^-1/(mol·cm^-2)] converted to cm^2/molecule,
    then to m^2/molecule.
    """
    if len(band_data["nu"]) == 0:
        return 1e-10  # floor

    nu    = band_data["nu"]
    S_ref = band_data["S"]
    gamma_air  = band_data["gamma_air"]
    gamma_self = band_data["gamma_self"]
    E_lower    = band_data["E_lower"]
    n_air      = band_data["n_air"]

    # Scale intensities to temperature T
    S_T = scale_line_intensity(S_ref, E_lower, nu, T, mol)

    # Lorentzian HWHM (pressure broadening) [cm^-1]
    gamma_L = ((gamma_air * (1.0 - X) + gamma_self * X)
               * p * (T0 / T) ** n_air)

    # Doppler HWHM [cm^-1]
    # gamma_D = (nu/c) * sqrt(2*kT*ln2/m)
    # = nu * sqrt(2*kT*ln2 / (m_mol * c^2))
    # In wavenumber units: gamma_D = nu * 3.581e-7 * sqrt(T/mol_mass)
    gamma_D = nu * 3.5810e-7 * np.sqrt(T / mol_mass)

    # Voigt HWHM (pseudo-Voigt approximation)
    # We sum line cross-sections; each line's integrated cross section = S_T
    # Voigt peak: sigma_peak = S_T / (sqrt(pi) * gamma_V)
    # For band average we integrate the Voigt profile over the band —
    # since lines are narrow, integral ~ S_T (the total line area)
    # So band kappa ~ n_mol * sum(S_T_i)  [converted to SI]

    # S_T is in cm^-1 / (molecule/cm^2) = cm^2/molecule (cross-section)
    # Convert to m^2/molecule: multiply by 1e-4
    sigma_per_molecule = S_T * 1.0e-4  # [m^2/molecule]

    # Number density of absorbing species [molecules/m^3]
    # n = X * p_Pa / (kB * T)
    p_Pa    = p * 101325.0          # atm → Pa
    n_total = p_Pa / (KB * T)       # total number density [molecules/m^3]
    n_mol   = X * n_total           # species number density [molecules/m^3]

    # Band-averaged absorption coefficient
    kappa = n_mol * np.sum(sigma_per_molecule)

    return max(kappa, 1e-10)


def generate_lookup_table(species_list: list, data_dir: Path, output_path: Path):
    """
    Main function: build kappa(T, p, X) lookup table for all species and bands.
    Each (T, p, X) grid point gets contributions from all species present.
    """
    print(f"\nGenerating lookup table: {output_path}")
    print(f"Grid: {len(T_GRID)} T × {len(P_GRID)} p × {len(X_GRID)} X = "
          f"{len(T_GRID)*len(P_GRID)*len(X_GRID):,} points")

    # Load all .par files
    all_band_data = {}
    for sp in species_list:
        mol_id = MOL_IDS[sp]
        par_file = data_dir / f"{sp}.par"
        print(f"\nLoading {sp} (mol_id={mol_id}):")
        all_band_data[sp] = load_par_in_bands(str(par_file), mol_id)

    # Write output header
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        header = (
            "# HITEMP band-averaged absorption coefficient lookup table\n"
            "# Generated by scripts/process_hitemp.py\n"
            f"# Species: {', '.join(species_list)}\n"
            f"# Bands: {[(b[0],b[1]) for b in BANDS]}\n"
            "#\n"
            "# Columns: T[K]  p[atm]  X[-]"
        )
        for bi, (nu_lo, nu_hi, label) in enumerate(BANDS):
            header += f"  kappaBand{bi}[1/m]({label})"
        header += "\n"
        f.write(header)

        total_points = len(T_GRID) * len(P_GRID) * len(X_GRID)
        count = 0

        for T in T_GRID:
            for p in P_GRID:
                for X in X_GRID:
                    kappa_bands = np.zeros(N_BANDS)

                    for sp in species_list:
                        mol_mass = MOL_MASS[sp]
                        for bi in range(N_BANDS):
                            kappa_bands[bi] += compute_band_kappa(
                                all_band_data[sp][bi], T, p, X, sp, mol_mass
                            )

                    row = f"{T:.1f}  {p:.4f}  {X:.5f}"
                    for k in kappa_bands:
                        row += f"  {k:.6e}"
                    f.write(row + "\n")
                    count += 1

                    if count % 500 == 0:
                        print(f"  {count}/{total_points} points done ...", end="\r")

        print(f"  {total_points}/{total_points} points done.         ")

    print(f"\nLookup table written to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", nargs="+", default=["H2O", "CO2", "CO"],
                        choices=["H2O", "CO2", "CO"],
                        help="Species to include (default: H2O CO2 CO)")
    parser.add_argument("--data-dir", default="irSignature/hitemp_data",
                        help="Directory containing .par files")
    parser.add_argument("--output", default="irSignature/hitemp_data/kappa_lookup.dat",
                        help="Output lookup table path")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    output_path = Path(args.output)

    print("=" * 60)
    print("HITEMP Band-Averaged Absorption Coefficient Generator")
    print("=" * 60)
    print(f"Species: {args.species}")
    print(f"Data dir: {data_dir}")
    print(f"Output: {output_path}")
    print(f"Bands:")
    for bi, (nu_lo, nu_hi, label) in enumerate(BANDS):
        print(f"  Band {bi}: {nu_lo}–{nu_hi} cm^-1  ({label})")

    generate_lookup_table(args.species, data_dir, output_path)


if __name__ == "__main__":
    main()
