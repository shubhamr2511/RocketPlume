#!/usr/bin/env python3
"""
extract_ir.py
-------------
Post-process OpenFOAM radiation results to extract IR signature at the sensor.

Reads ILambda_band{b}_ray{d} fields from the OpenFOAM case and:
  1. Identifies ray directions pointing toward the sensor patch
  2. Integrates intensity over the sensor area and solid angle
  3. Reports total received power per band [W] and spectral distribution
  4. Plots spectral radiant intensity vs wavelength

Usage:
    python scripts/extract_ir.py \
        --case irSignature \
        --time 1 \
        --sensor outlet \
        --nBands 3 \
        --nRays 64

Output:
    ir_signature.txt   — per-band power summary
    ir_spectrum.png    — spectral plot
"""

import argparse
import os
import re
import sys
import numpy as np
from pathlib import Path


# ── Band definitions (must match radiationProperties) ─────────────────────────
BANDS = [
    (2000.0, 3333.0, "MWIR (3-5 µm)"),
    (1000.0, 2000.0, "MIR (5-10 µm)"),
    ( 714.0, 1250.0, "LWIR (8-14 µm)"),
]


def wavenumber_to_wavelength(nu: float) -> float:
    """Convert wavenumber [cm^-1] to wavelength [µm]."""
    return 10000.0 / nu


def band_center_wavelength(nu_lo: float, nu_hi: float) -> float:
    """Band center wavelength [µm]."""
    return wavenumber_to_wavelength(0.5 * (nu_lo + nu_hi))


# ── OpenFOAM field reader ─────────────────────────────────────────────────────

def read_of_scalar_field(field_path: Path) -> np.ndarray:
    """
    Read an OpenFOAM ASCII volScalarField internalField.
    Returns 1D numpy array of values.
    """
    if not field_path.exists():
        return None

    values = []
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
                    values.append(float(stripped))
                except ValueError:
                    pass

    return np.array(values) if values else None


def read_boundary_patch_field(field_path: Path, patch_name: str) -> np.ndarray:
    """
    Read values on a named boundary patch from an OpenFOAM field file.
    Returns face-averaged array or None if patch not found.
    """
    if not field_path.exists():
        return None

    content = field_path.read_text()

    # Find the boundary patch section
    pattern = rf"{re.escape(patch_name)}\s*\{{[^}}]*?\}}"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    patch_block = match.group(0)

    # Extract uniform value
    uni_match = re.search(r"value\s+uniform\s+([+-]?[\d.eE+-]+)", patch_block)
    if uni_match:
        return np.array([float(uni_match.group(1))])

    # Extract nonuniform list
    list_match = re.search(r"value\s+nonuniform List<scalar>\s+\d+\s*\(\s*([\d\s.eE+-]+)\)",
                            patch_block, re.DOTALL)
    if list_match:
        return np.fromstring(list_match.group(1), sep="\n")

    return None


def read_ray_directions(case_dir: Path, n_rays: int) -> np.ndarray:
    """
    Read discrete ordinate ray directions from the radiation properties
    or reconstruct from nPhi/nTheta.
    Returns array of shape (nRays, 3).
    """
    n_phi   = 4
    n_theta = 4
    # Try to read from radiationProperties
    rad_props = case_dir / "constant" / "radiationProperties"
    if rad_props.exists():
        content = rad_props.read_text()
        m_phi   = re.search(r"nPhi\s+(\d+)", content)
        m_theta = re.search(r"nTheta\s+(\d+)", content)
        if m_phi:   n_phi   = int(m_phi.group(1))
        if m_theta: n_theta = int(m_theta.group(1))

    # Reconstruct ray directions (must match myFvDOM::buildQuadrature())
    dirs   = []
    omegas = []
    d_phi   = 2 * np.pi / (4 * n_phi)
    d_theta = np.pi / n_theta

    for i_phi in range(4 * n_phi):
        phi = (i_phi + 0.5) * d_phi
        for i_theta in range(n_theta):
            theta = (i_theta + 0.5) * d_theta
            dirs.append([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            omegas.append(np.sin(theta) * d_theta * d_phi)

    return np.array(dirs), np.array(omegas)


# ── Sensor normal (assumed pointing in -x direction for outlet patch) ──────────

def get_patch_normal(patch_name: str) -> np.ndarray:
    """
    Return the inward normal of the sensor patch.
    For the axisymmetric wedge case, the outlet normal is (-1, 0, 0).
    Modify this if using a different geometry.
    """
    normals = {
        "outlet":  np.array([-1.0, 0.0, 0.0]),   # plume flows in +x
        "sensor":  np.array([-1.0, 0.0, 0.0]),
        "inlet":   np.array([1.0,  0.0, 0.0]),
    }
    return normals.get(patch_name, np.array([-1.0, 0.0, 0.0]))


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_ir_signature(case_dir: Path, time_name: str, sensor_patch: str,
                          n_bands: int, n_rays: int) -> dict:
    """
    Extract per-band IR intensity at the sensor patch.

    Returns dict:
        "power_per_band": [W] power in each band
        "intensity_per_band": [W/m^2/sr] average intensity per band
    """
    time_dir = case_dir / time_name
    ray_dirs, omegas = read_ray_directions(case_dir, n_rays)

    sensor_normal = get_patch_normal(sensor_patch)

    # Identify rays pointing toward sensor (dot product with inward normal > 0)
    inward_rays = []
    for d, (rdir, omega) in enumerate(zip(ray_dirs, omegas)):
        cos_theta = np.dot(rdir, sensor_normal)
        if cos_theta > 0:
            inward_rays.append((d, cos_theta, omega))

    if not inward_rays:
        print("WARNING: No ray directions pointing toward sensor patch!")
        print("         Check sensor patch name and geometry.")
        return None

    print(f"  {len(inward_rays)} of {n_rays} rays directed toward '{sensor_patch}'")

    power_per_band     = np.zeros(n_bands)
    intensity_per_band = np.zeros(n_bands)

    for b in range(n_bands):
        I_total = 0.0
        n_rays_used = 0

        for (d, cos_theta, omega) in inward_rays:
            field_name = f"ILambda_band{b}_ray{d}"
            field_path = time_dir / field_name

            # Try patch values first
            patch_vals = read_boundary_patch_field(field_path, sensor_patch)

            if patch_vals is None:
                # Fall back to internal field (less accurate)
                patch_vals = read_of_scalar_field(field_path)
                if patch_vals is None:
                    continue
                # Use cells near outlet (last 5% of cells)
                patch_vals = patch_vals[int(0.95 * len(patch_vals)):]

            # Integrate: dI = I * cos(theta) * dOmega
            I_contrib = np.mean(patch_vals) * cos_theta * omega
            I_total  += I_contrib
            n_rays_used += 1

        intensity_per_band[b] = I_total  # [W/m^2/sr] (approximate)
        # Power = intensity * area (assume unit area for now)
        power_per_band[b] = I_total  # Will scale by actual patch area if available

        nu_lo, nu_hi, label = BANDS[b] if b < len(BANDS) else (0, 0, "unknown")
        print(f"  Band {b} ({label}): I = {I_total:.4e} W/(m²·sr)  [{n_rays_used} rays]")

    return {
        "power_per_band":     power_per_band,
        "intensity_per_band": intensity_per_band,
    }


def plot_spectrum(results: dict, output_path: Path):
    """Plot spectral radiant intensity vs band center wavelength."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    n_bands = len(results["intensity_per_band"])
    wavelengths = []
    intensities = []
    labels      = []
    colors      = ["#E74C3C", "#3498DB", "#2ECC71"]

    for b in range(n_bands):
        if b < len(BANDS):
            nu_lo, nu_hi, label = BANDS[b]
            lam_center = band_center_wavelength(nu_lo, nu_hi)
            lam_lo     = wavenumber_to_wavelength(nu_hi)
            lam_hi     = wavenumber_to_wavelength(nu_lo)
            wavelengths.append((lam_center, lam_lo, lam_hi))
            intensities.append(results["intensity_per_band"][b])
            labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ((lam_c, lam_lo, lam_hi), I, label, color) in \
            enumerate(zip(wavelengths, intensities, labels, colors)):
        width = lam_hi - lam_lo
        ax.bar(lam_c, I, width=width * 0.8, color=color, alpha=0.8,
               label=f"Band {i}: {label}", edgecolor="black", linewidth=0.5)
        ax.text(lam_c, I * 1.05, f"{I:.2e}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xlabel("Wavelength [µm]", fontsize=12)
    ax.set_ylabel("Radiant Intensity [W/(m²·sr)]", fontsize=12)
    ax.set_title("IR Signature Spectral Distribution — Aircraft Exhaust Plume", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Add atmospheric windows
    ax.axvspan(3.0, 5.0, alpha=0.08, color="red",  label="MWIR window")
    ax.axvspan(8.0, 14.0, alpha=0.08, color="blue", label="LWIR window")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Spectrum plot saved: {output_path}")
    plt.close()


def write_text_report(results: dict, output_path: Path, case_name: str):
    """Write a plain-text IR signature summary report."""
    lines = [
        "=" * 60,
        "IR SIGNATURE ANALYSIS REPORT",
        f"Case: {case_name}",
        "=" * 60,
        "",
        "Band-wise Received Intensity at Sensor:",
        "-" * 40,
    ]

    total_I = 0.0
    for b, (nu_lo, nu_hi, label) in enumerate(BANDS):
        I = results["intensity_per_band"][b]
        lam_lo = wavenumber_to_wavelength(nu_hi)
        lam_hi = wavenumber_to_wavelength(nu_lo)
        lines.append(f"Band {b}: {label}")
        lines.append(f"  Wavenumber: {nu_lo:.0f}–{nu_hi:.0f} cm^-1")
        lines.append(f"  Wavelength: {lam_lo:.1f}–{lam_hi:.1f} µm")
        lines.append(f"  Intensity:  {I:.4e} W/(m²·sr)")
        lines.append("")
        total_I += I

    lines += [
        "-" * 40,
        f"Total Intensity: {total_I:.4e} W/(m²·sr)",
        "",
        "Band Fractions:",
    ]
    for b in range(len(BANDS)):
        frac = results["intensity_per_band"][b] / total_I * 100 if total_I > 0 else 0
        lines.append(f"  Band {b}: {frac:.1f}%")

    lines += [
        "",
        "Notes:",
        "  - Intensity units: W per m² of sensor per steradian",
        "  - Multiply by sensor area [m²] to get power [W]",
        "  - MWIR (3-5 µm) is primary band for IR-guided missiles",
        "=" * 60,
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    for line in lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case",   default="irSignature",
                        help="OpenFOAM case directory")
    parser.add_argument("--time",   default="1",
                        help="Time directory name (default: 1)")
    parser.add_argument("--sensor", default="outlet",
                        help="Sensor patch name (default: outlet)")
    parser.add_argument("--nBands", type=int, default=3,
                        help="Number of spectral bands")
    parser.add_argument("--nRays",  type=int, default=64,
                        help="Total number of ray directions (4*nPhi*nTheta)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files")
    args = parser.parse_args()

    case_dir   = Path(args.case)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("IR Signature Extraction")
    print("=" * 60)
    print(f"Case:   {case_dir}")
    print(f"Time:   {args.time}")
    print(f"Sensor: {args.sensor}")
    print(f"Bands:  {args.nBands}, Rays: {args.nRays}")
    print()

    results = extract_ir_signature(
        case_dir, args.time, args.sensor, args.nBands, args.nRays
    )

    if results is None:
        print("ERROR: Could not extract IR signature.")
        sys.exit(1)

    # Write outputs
    report_path  = output_dir / "ir_signature.txt"
    spectrum_path = output_dir / "ir_spectrum.png"

    write_text_report(results, report_path, args.case)
    plot_spectrum(results, spectrum_path)


if __name__ == "__main__":
    main()
