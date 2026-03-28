#!/usr/bin/env python3
"""
validate_gray.py
----------------
Analytical validation of the gray RTE solver against the optically thin limit.

In the optically thin limit (kappa * L << 1):
    I(L) ≈ I(0) + kappa * Ib * L

In the optically thick limit (kappa * L >> 1):
    I ≈ Ib  (local blackbody emission)

This script:
  1. Computes expected intensity along a ray path through the plume
  2. Reads the OpenFOAM ILambda field
  3. Compares and reports relative error

Usage:
    python scripts/validate_gray.py --case irSignature --band 0 --ray 0
"""

import argparse
import numpy as np
from pathlib import Path


SIGMA = 5.67037e-8   # Stefan-Boltzmann constant [W/m^2/K^4]


def planck_total(T: float) -> float:
    """Total blackbody emission power [W/m^2]: Eb = sigma * T^4"""
    return SIGMA * T**4


def planck_intensity(T: float) -> float:
    """Isotropic blackbody intensity [W/m^2/sr]: Ib = Eb / pi"""
    return planck_total(T) / np.pi


def fractional_blackbody(lam_T: float) -> float:
    """Fractional blackbody F(0→lambda*T). Chang-Rhee series."""
    if lam_T <= 0:
        return 0.0
    C2 = 14387.77   # µm·K
    x  = C2 / lam_T
    if x > 100:
        return 0.0
    F = 0.0
    for n in range(1, 21):
        en = np.exp(-n * x)
        if en < 1e-15:
            break
        F += en / n * (x**3 + 3*x**2/n + 6*x/n**2 + 6/n**3)
    F *= 15 / np.pi**4
    return np.clip(F, 0, 1)


def planck_band_intensity(T: float, nu_lo: float, nu_hi: float) -> float:
    """Band Planck intensity [W/m^2/sr] for band [nu_lo, nu_hi] cm^-1."""
    lam_lo = 10000.0 / nu_hi    # µm (note reversal)
    lam_hi = 10000.0 / nu_lo
    F_hi = fractional_blackbody(lam_hi * T)
    F_lo = fractional_blackbody(lam_lo * T)
    return (F_hi - F_lo) * planck_total(T) / np.pi


def integrate_rte_1d(x_arr: np.ndarray, kappa_arr: np.ndarray,
                      Ib_arr: np.ndarray, I0: float = 0.0) -> np.ndarray:
    """
    Integrate the 1D RTE along a ray path using the formal solution:
        I(s) = I(0)*exp(-tau(0,s)) + integral_0^s kappa(s')*Ib(s')*exp(-tau(s',s)) ds'

    Uses the trapezoidal rule for the path integral.
    Returns I at each x position.
    """
    n    = len(x_arr)
    I    = np.zeros(n)
    I[0] = I0

    for i in range(1, n):
        dx = x_arr[i] - x_arr[i-1]
        # Optical depth increment
        dtau = 0.5 * (kappa_arr[i] + kappa_arr[i-1]) * dx

        # Exponential decay + source term
        trans = np.exp(-dtau)
        source = 0.5 * (kappa_arr[i] * Ib_arr[i] + kappa_arr[i-1] * Ib_arr[i-1])
        I[i] = I[i-1] * trans + source * dx

    return I


def run_validation(case_dir: Path, band_idx: int, ray_idx: int):
    """Compare OpenFOAM result with analytical 1D RTE integration."""
    print(f"\nValidation: case={case_dir}, band={band_idx}, ray={ray_idx}")

    BANDS = [
        (2000.0, 3333.0, "MWIR"),
        (1000.0, 2000.0, "MIR"),
        ( 714.0, 1250.0, "LWIR"),
    ]
    nu_lo, nu_hi, label = BANDS[band_idx]

    # ── Synthetic plume profile for analytical solution ──
    L       = 0.4         # plume length [m]
    T_peak  = 1200.0      # K at nozzle
    T_amb   = 288.0       # K ambient
    kappa0  = 5.0         # typical absorption coefficient [1/m]

    n_pts = 200
    x     = np.linspace(0, L, n_pts)

    # Gaussian temperature profile along centerline
    T_x   = T_amb + (T_peak - T_amb) * np.exp(-5.0 * x / L)
    kappa = kappa0 * (T_x - T_amb) / (T_peak - T_amb)

    Ib_band = np.array([planck_band_intensity(T, nu_lo, nu_hi) for T in T_x])

    # Analytical integration
    I_analytic = integrate_rte_1d(x, kappa, Ib_band, I0=0.0)

    # Optically thin approximation (for comparison)
    tau_total = np.trapz(kappa, x)
    I_thin    = np.trapz(kappa * Ib_band, x)

    print(f"\n  Band: {label} ({nu_lo:.0f}–{nu_hi:.0f} cm^-1)")
    print(f"  Optical depth tau = {tau_total:.4f}")
    print(f"  Optically {'thin' if tau_total < 1 else 'thick'} regime (tau {'<' if tau_total < 1 else '>'} 1)")
    print(f"\n  Analytical 1D RTE solution (centerline ray):")
    print(f"    I(inlet)  = {I_analytic[0]:.4e} W/(m²·sr)")
    print(f"    I(outlet) = {I_analytic[-1]:.4e} W/(m²·sr)")
    print(f"    I_thin    = {I_thin:.4e} W/(m²·sr)  (thin approximation)")
    print(f"    I_thick   = {planck_band_intensity(T_peak, nu_lo, nu_hi):.4e} W/(m²·sr)  (thick limit)")

    # ── Try to read OpenFOAM result ──
    time_dir   = case_dir / "1"
    field_name = f"ILambda_band{band_idx}_ray{ray_idx}"
    field_path = time_dir / field_name

    if not field_path.exists():
        print(f"\n  OpenFOAM field {field_path} not found.")
        print("  Run the solver first: myRadiationSolver")
        print("\n  Analytical result can still be used for solver validation.")
        return I_analytic[-1]

    # Read OpenFOAM field
    vals = []
    reading = False
    with open(field_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "(":
                reading = True
                continue
            if reading:
                if stripped in (")", ");"):
                    break
                try:
                    vals.append(float(stripped))
                except ValueError:
                    pass

    if not vals:
        print("  Could not read OpenFOAM field values.")
        return None

    I_of = np.array(vals)
    I_of_outlet = I_of[-1]   # last cell (outlet)

    rel_error = abs(I_of_outlet - I_analytic[-1]) / (I_analytic[-1] + 1e-20)

    print(f"\n  OpenFOAM result:")
    print(f"    I(outlet) = {I_of_outlet:.4e} W/(m²·sr)")
    print(f"    Analytical = {I_analytic[-1]:.4e} W/(m²·sr)")
    print(f"    Relative error = {rel_error*100:.2f}%")

    if rel_error < 0.05:
        print("  ✓ PASS: Error < 5%")
    elif rel_error < 0.15:
        print("  ~ ACCEPTABLE: Error < 15% (check mesh resolution)")
    else:
        print("  ✗ FAIL: Error > 15% — check kappa mapping and BCs")

    return I_of_outlet


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case",  default="irSignature")
    parser.add_argument("--band",  type=int, default=0)
    parser.add_argument("--ray",   type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("Gray RTE Analytical Validation")
    print("=" * 60)

    run_validation(Path(args.case), args.band, args.ray)

    print("\nFor full validation run all bands:")
    for b in range(3):
        print(f"  python scripts/validate_gray.py --case {args.case} --band {b} --ray 0")


if __name__ == "__main__":
    main()
