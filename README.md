# Bayfol MNIST phase-mask experiment

A small TensorFlow experiment for asking a concrete question: **how much does a single phase mask help the MNIST digit classifier as we give the mask more spatial phase cells, and what does a 16 µm Bayfol volume do to the optical channels?**

This is an adaptation of Lin et al.'s 2018 diffractive MNIST task to one phase screen in a LETO-to-Bayfol setup. The paper used **five** transmissive diffractive layers, 55,000 MNIST training images, 5,000 validation images, and 10 detector regions. This project keeps the dataset, amplitude-coded input, and ten-detector decision rule, and trains a **single phase-only screen** because one Bayfol film is one recording volume. It is a different, shallower optical classifier; it does not claim to reproduce their five-layer result. [Paper](https://doi.org/10.1126/science.aat8084).

The film is modeled in two deliberately separate ways:

- `run.py` trains a pixelated phase screen and numerically propagates MNIST fields to ten detector regions. `cells × cells` is the chosen screen parameter budget.
- `capacity.py` reports that screen budget and estimates the angular half-power width of **one weak volume grating** for a chosen recording geometry. The 16 µm thickness changes this Bragg-selectivity screen. It does not by itself determine the count of independent spatial phase values.
- `exposure.py` uses a trained one-screen phase map as the desired object field, then computes a candidate 1920×1080 LETO drive image to record that field against a separate coherent reference beam.

**No multiplexing is implemented.** The angular-width calculation is a first-order screen, not a measured capacity, a coupled-wave efficiency prediction, or a material recording/readout simulation. The actual number of usable independent weights remains unknown until channel cross-talk and the phase-map readout are measured in the film.

## Start here

Python 3.11 is the supported reproducible environment. From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a short training pass first. TensorFlow downloads the standard MNIST dataset automatically:

```bash
python run.py --epochs 3 --train 2000 --test 1000
```

Then train the configured one-screen model on the full 55,000 / 5,000 / 10,000 split:

```bash
python run.py --epochs 5
```

Each run prints its output folder and saves its JSON configuration and metrics, the best validation phase map, sample predictions, and a plot under `runs/<timestamp>/`. Runs are ignored by Git so the repository stays small. To inspect saved results in the browser:

```bash
streamlit run app.py
```

To see how screen resolution changes the nominal parameter budget and the film's modeled angular selectivity:

```bash
python capacity.py --output runs/capacity-screen.json
```

Change the recording angles explicitly when they are measured for your setup:

```bash
python capacity.py --object-angle-deg -48.16 --reference-angle-deg 48.16
```

## Candidate LETO recording image

After training a **one-layer** run, request the object-arm phase image:

```bash
python exposure.py --run runs/full-one-screen --steps 100 --focal-mm 200
```

The output is saved in `runs/full-one-screen/exposure/`. The default image assumes the ideal mapping `gray = phase × 256 / (2π)`. If you have a measured 256-value gray-to-phase LUT for your LETO, pass it with `--lut path/to/measured_phase_lut.npy`.

This image is a candidate recording pattern. Displaying it on the LETO alone does not write or validate a useful hologram: the film also needs the matching coherent reference, exposure and material response. Use native 1920×1080 pixels and a device configuration with the measured wavelength-specific phase calibration. The simulation defaults (532 nm, 200 mm focal length, 0.8192 mm illuminated film aperture) are example values to replace with your measured optical setup.

## What the capacity numbers mean

For `k` phase cells on each side, the classifier has `k²` trainable phase values. A common phase added everywhere does not change detector intensities, leaving at most `k² − 1` observable phase degrees of freedom in this one-screen model. This is a digital model count, not proof that Bayfol records or reconstructs each value independently.

The 16 µm thickness enters the volume-grating response through the longitudinal phase-mismatch integral, approximated here by a normalized `sinc²` efficiency for a weak uniform grating. This gives an angular selectivity estimate for one grating at the supplied wavelength, refractive index, and object/reference angles. Actual efficiency and cross-talk depend on measured index modulation, polarization, exposure, shrinkage, surface geometry, and the chosen readout. Thickness alone cannot tell us how many classifier weights work.

The useful experiment is to train the classifier at increasing `--cells`, produce a candidate phase image, record/read it under a fixed geometry, and measure detector leakage and digit accuracy. That measured curve—not the nominal cell count or the weak-grating estimate—will establish a practical limit for this setup.

## Physical starting assumptions

`config.json` puts a square 0.8192 mm active phase aperture on a 128×128 simulation grid. This is an intentionally small first model, not the full LETO illuminated aperture. The screen samples grayscale MNIST as input field amplitude, uses scalar paraxial Fresnel propagation, and assigns the ten output classes to ten detector regions. The film is represented as one ideal phase screen during classifier training; exposure/readout imperfections enter only through future measurements. The exposure utility computes the LETO pattern for a target field at the film and does not simulate the film's index modulation.

If your measured film active thickness, refractive index, illumination aperture, relay distances, laser wavelength, or recording angles differ, edit `config.json` or pass the relevant command options before interpreting a result. Do not compare a small training run's accuracy directly with the paper's five-layer result.

## Code map

| File | Purpose |
|---|---|
| `optics.py` | Differentiable Fresnel propagation, phase screen, input encoding, detector readout |
| `run.py` | Dataset split, TensorFlow training, held-out accuracy and saved run artifacts |
| `capacity.py` | Nominal phase-cell budget plus one-grating angular selectivity screen |
| `exposure.py` | LETO phase-only inverse design for one trained screen |
| `app.py` | Local browser interface to inspect runs and change the cell-count experiment |
| `config.json` | Editable starting geometry and numerical settings |

## References

- X. Lin et al., “All-optical machine learning using diffractive deep neural networks,” *Science* 361, 1004–1008 (2018), [doi:10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084).
- Covestro, [Bayfol HX200 product data](https://solutions.covestro.com/en/products/bayfol/bayfol-hx200_000000000086212099).

## License

Project code is MIT-licensed. MNIST is downloaded from the TensorFlow dataset service when needed and is not bundled here.
