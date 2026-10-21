"""First-order Bayfol capacity screens; outputs assumptions, never a claim."""
import argparse
import json
from pathlib import Path

import numpy as np


def wavevector(external_angle_deg, wavelength_m, refractive_index):
    """2-D wavevector inside film; angle is signed from the +z normal in air."""
    theta = np.deg2rad(external_angle_deg)
    k0 = 2 * np.pi / wavelength_m
    k = refractive_index * k0
    kx = k0 * np.sin(theta)
    kz = np.sign(np.cos(theta)) * np.sqrt(max(k * k - kx * kx, 0.0))
    return np.array([kx, kz])


def relative_weak_grating_efficiency(
    read_angle_deg,
    object_angle_deg,
    reference_angle_deg,
    wavelength_m,
    thickness_m,
    refractive_index,
):
    """sinc^2 longitudinal-mismatch screen normalized to on-Bragg response."""
    obj = wavevector(object_angle_deg, wavelength_m, refractive_index)
    reference = wavevector(reference_angle_deg, wavelength_m, refractive_index)
    read = wavevector(read_angle_deg, wavelength_m, refractive_index)
    grating = obj - reference
    scattered = read + grating
    k = 2 * np.pi * refractive_index / wavelength_m
    if abs(scattered[0]) >= k:
        return 0.0
    kz_out = np.sqrt(k * k - scattered[0] * scattered[0])
    mismatch = scattered[1] - kz_out
    # np.sinc(x) = sin(pi*x)/(pi*x).
    return float(np.sinc(mismatch * thickness_m / (2 * np.pi)) ** 2)


def angular_fwhm_deg(
    object_angle_deg,
    reference_angle_deg,
    wavelength_m,
    thickness_m,
    refractive_index,
):
    """Find the contiguous half-power angular width around one Bragg peak."""
    offsets = np.linspace(-15.0, 15.0, 60001)
    efficiencies = np.array(
        [
            relative_weak_grating_efficiency(
                reference_angle_deg + offset,
                object_angle_deg,
                reference_angle_deg,
                wavelength_m,
                thickness_m,
                refractive_index,
            )
            for offset in offsets
        ]
    )
    peak = int(np.argmax(efficiencies))
    left = peak
    right = peak
    while left > 0 and efficiencies[left] >= 0.5:
        left -= 1
    while right < len(offsets) - 1 and efficiencies[right] >= 0.5:
        right += 1
    if left == 0 or right == len(offsets) - 1:
        return None
    return float(offsets[right] - offsets[left])


def phase_budget(aperture_mm, cells):
    """Nominal phase values in one pixelated screen, after removing global phase."""
    if aperture_mm <= 0 or cells < 2:
        raise ValueError("aperture_mm must be positive and cells at least 2")
    return {
        "cells_per_axis": cells,
        "phase_values": cells * cells,
        "intensity_observable_phase_degrees_of_freedom": cells * cells - 1,
        "cell_pitch_um": aperture_mm * 1000 / cells,
        "meaning": "Numerical phase-screen budget; not measured independent Bayfol weights.",
    }


def angular_channel_screen(count, spacing_deg, object_angle_deg, reference_angle_deg,
                           wavelength_m, thickness_m, refractive_index):
    """Weak-grating angular response for separately recorded reference-angle channels."""
    if count < 1 or spacing_deg <= 0:
        raise ValueError("channels must be >=1 and angle spacing must be positive")
    offsets = (np.arange(count) - (count - 1) / 2) * spacing_deg
    refs = reference_angle_deg + offsets
    response = np.zeros((count, count), dtype=float)
    for read_idx, read in enumerate(refs):
        for recorded_idx, recorded_ref in enumerate(refs):
            response[read_idx, recorded_idx] = relative_weak_grating_efficiency(
                read, object_angle_deg, recorded_ref, wavelength_m,
                thickness_m, refractive_index)
    diagonal = np.diag(response)
    normalized = response / np.maximum(diagonal[:, None], 1e-12)
    return {
        "channel_count": count,
        "reference_angle_spacing_deg": spacing_deg,
        "reference_angles_external_deg": refs.tolist(),
        "weak_grating_response_matrix": response.tolist(),
        "off_channel_to_selected_channel_ratio": normalized.tolist(),
        "scope": "Idealized weak-grating angular selectivity only; not Bayfol recording capacity or measured cross-talk.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="runs/capacity-screen.json")
    parser.add_argument("--cells", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--object-angle-deg", type=float, default=-48.16)
    parser.add_argument("--reference-angle-deg", type=float, default=48.16)
    parser.add_argument("--angular-channels", type=int, default=3)
    parser.add_argument("--angle-spacing-deg", type=float, default=1.0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    thickness = config["film_thickness_um"] * 1e-6
    wavelength = config["wavelength_nm"] * 1e-9
    index = config["film_index"]
    result = {
        "scope": "One-grating angular response plus an idealized reference-angle channel screen; not a multiplexing capacity or storage-capacity prediction.",
        "assumptions": {
            "film_thickness_um": config["film_thickness_um"],
            "wavelength_nm": config["wavelength_nm"],
            "unrecorded_refractive_index": index,
            "object_angle_external_deg": args.object_angle_deg,
            "reference_angle_external_deg": args.reference_angle_deg,
            "model": "Uniform weak-grating sinc-squared longitudinal mismatch; real strong-grating efficiency needs coupled-wave model and measurements.",
        },
        "single_mask_phase_budgets": [
            phase_budget(config["aperture_mm"], cells) for cells in args.cells
        ],
        "weak_grating_angular_fwhm_external_deg": angular_fwhm_deg(
            args.object_angle_deg,
            args.reference_angle_deg,
            wavelength,
            thickness,
            index,
        ),
        "angular_multiplexing_screen": angular_channel_screen(
            args.angular_channels, args.angle_spacing_deg, args.object_angle_deg,
            args.reference_angle_deg, wavelength, thickness, index),
        "interpretation": "Cell count controls the classifier mask's numerical phase degrees of freedom. Thickness mainly changes volume-grating angular and wavelength selectivity. The two counts are not interchangeable.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
