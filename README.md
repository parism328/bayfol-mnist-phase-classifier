# Bayfol MNIST phase-mask experiment

A small TensorFlow experiment for asking a concrete question: **how much does a single phase mask help the MNIST digit classifier as we give the mask more spatial phase cells, and what does a 16 µm Bayfol volume do to the optical channels?**

This is an adaptation of Lin et al.'s 2018 diffractive MNIST task to **one physical diffractive layer** in a LETO-to-Bayfol setup. The paper used five transmissive diffractive layers, 55,000 MNIST training images, 5,000 validation images, and 10 detector regions. This project keeps the dataset, amplitude-coded input, and ten-detector decision rule, but models one phase-only screen. [Paper](https://doi.org/10.1126/science.aat8084).

The intended physical configuration is **angular multiplexing in that one Bayfol film**: record several holograms in the same 16 µm volume, each with a different reference-beam angle, then select a stored hologram by changing the readout angle. These are multiple angular channels in one recording volume, not multiple film layers. The optional `multiplex.py` path creates separate numerical classifiers and candidate LETO patterns for these channels; it does not simulate the physical film recording or readout.

The film is modeled in two deliberately separate ways:

- `run.py` trains a pixelated phase screen and numerically propagates MNIST fields to ten detector regions. `cells × cells` is the chosen screen parameter budget.
- `capacity.py` reports the numerical screen budget, estimates the angular half-power width of one weak volume grating, and calculates a tentative response matrix for a chosen set of reference-angle channels. The 16 µm thickness affects this Bragg-selectivity estimate. It does not by itself determine independent spatial phase values or usable multiplexed hologram count.
- `exposure.py` uses a trained one-screen phase map as the desired object field, then computes a candidate 1920×1080 LETO drive image to record that field against a separate coherent reference beam.

`multiplex.py` adds an optional numerical channel-set workflow: it independently trains one single-screen MNIST classifier per angular channel and generates one candidate LETO object-arm pattern per trained mask. It assigns a different reference angle to each channel and reports the weak-grating response matrix. This does not simulate Bayfol recording competition, fringes, exposure, or reconstructed classification. The actual number of usable angular channels and independent spatial weights remains unknown until recording competition, channel cross-talk, and phase-map readout are measured in the film.

## Start here

Python 3.11 is the supported reproducible environment. From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-training.txt
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

## Public hosted viewer

The app can be deployed from this public GitHub repository with Streamlit Community Cloud. The committed `demo_artifacts/` folder supplies a small read-only example so the hosted viewer has patterns and classifier results when it starts. Community Cloud runs the Python app from the repository; GitHub Pages only serves static HTML/JavaScript and cannot run this Streamlit/Python simulation. To deploy, sign in at [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select `parism328/bayfol-mnist-phase-classifier`, branch `main`, and entrypoint `app.py`. The root `requirements.txt` contains only viewer dependencies; `requirements-training.txt` adds TensorFlow for local training.

To see how screen resolution changes the nominal parameter budget, and estimate angular selectivity for three recordings in one volume:

```bash
python capacity.py --angular-channels 3 --angle-spacing-deg 1.0 --output runs/capacity-screen.json
```

Change the recording angles explicitly when they are measured for your setup:

```bash
python capacity.py --object-angle-deg -48.16 --reference-angle-deg 48.16
```

The output includes a weak-grating response matrix. Rows correspond to readout-angle choices and columns to recordings with different reference angles. Its off-diagonal terms are a rough angular-leakage screen. Sweep channel count and spacing, then compare this estimate with measured diffraction efficiency versus readout angle for individual and multiplexed recordings.

In the angle controls, the object-beam angle is held fixed and the reference-beam angle is stepped symmetrically around the chosen center. These values label the channels and feed the weak-grating estimate; they do not steer hardware or change the generated LETO pixel pattern. The present estimate assumes a simplified same-forward-direction wavevector geometry. It does not model counter-propagating waves in a reflection hologram, so do not use its channel-spacing numbers to set the reflective Bayfol experiment.

## Optional angular-multiplexed classifier set

Select **Angular-multiplexed channel set** in the Streamlit interface. Or build the same artifact set from the project terminal:

```bash
python multiplex.py --channels 3 --angle-spacing-deg 2.5 --object-angle-deg -48.16 --reference-angle-deg 48.16 --epochs 3 --train 2000 --test 1000 --exposure-steps 20
```

This trains three independent phase masks, then writes one LETO object-arm candidate per channel under `runs/multiplex-<timestamp>/channel-NN/exposure/`. The `manifest.json` lists each file and its assigned reference angle. The patterns target the same film patch; during a physical recording, each would be displayed while its matching reference beam overlaps the film. The files do not contain reference fringes or provide exposure dose settings. Use a measured LETO phase LUT with `--lut path/to/measured_phase_lut.npy` before treating pixel values as phase commands. Short training settings are for a quick demo; increase data and epochs for a meaningful classifier study.

## Candidate LETO recording image

After training a **one-layer** run, request the object-arm phase image:

```bash
python exposure.py --run runs/full-one-screen --steps 100 --focal-mm 200
```

The output is saved in `runs/full-one-screen/exposure/`. `SLM_IDEAL_LUT.png` is a 1920×1080 phase-only LETO drive image. In the current example it is inverse-designed to create the learned 16×16 phase distribution across one 0.8192×0.8192 mm square at the modeled lens's back focal plane. That corresponds to 51.2 µm per learned phase cell. It targets **one continuous film patch**, not one hogel in a tiled hologram array. The target patch is represented by about 95×53 samples on the modeled focal-plane grid; this sampling is anisotropic because the SLM is 1920×1080. The PNG gray values currently use the ideal mapping `gray = phase × 256 / (2π)`. If you have a measured 256-value gray-to-phase LUT for your LETO, pass it with `--lut path/to/measured_phase_lut.npy`.

The PNG is the LETO **object-arm phase drive**, not the Bayfol fringe pattern and not a hogel tile. `drive_phase_radians.npy` stores the phase requested at each of the 1920×1080 LETO pixels after applying the selected gray-to-phase LUT; the viewer displays this array as a radians heatmap and lets you inspect individual pixel values. `phase_radians.npy` stores the continuous pre-quantization inverse-design phase. During recording, the object field must overlap the separate coherent reference beam in the target film patch; their interference writes a continuous volume hologram there. The export does not calculate the reference fringes, exposure dose, or material response, so the image is not hardware-ready until its phase LUT, illuminated aperture, focal length, wavelength, and recording geometry match measured values. The defaults (532 nm, 200 mm focal length, 0.8192 mm target aperture) are illustrative.

## What the capacity numbers mean

For `k` phase cells on each side, the classifier has `k²` trainable phase values. A common phase added everywhere does not change detector intensities, leaving at most `k² − 1` observable phase degrees of freedom in this one-screen model. This is a digital model count, not proof that Bayfol records or reconstructs each value independently.

The 16 µm thickness enters the volume-grating response through the longitudinal phase-mismatch integral, approximated here by a normalized `sinc²` efficiency for a weak uniform grating. This gives an angular selectivity estimate for one grating at the supplied wavelength, refractive index, and object/reference angles. Actual efficiency and cross-talk depend on measured index modulation, polarization, exposure, shrinkage, surface geometry, and the chosen readout. Thickness alone cannot tell us how many classifier weights work.

For the single-channel experiment, train at increasing `--cells`, produce a candidate phase image, record/read it, and measure detector leakage and digit accuracy. For angular multiplexing, separately record known test holograms at stepped reference angles, measure each reconstruction at every readout angle, and form an experimental cross-talk matrix. Only then choose a usable angle spacing and channel count. That measured curve—not the nominal cell count or weak-grating estimate—will establish practical limits for this setup.

## Physical starting assumptions

`config.json` puts a square 0.8192 mm active phase aperture on a 128×128 simulation grid. This is an intentionally small first model, not the full LETO illuminated aperture. The screen samples grayscale MNIST as input field amplitude, uses scalar paraxial Fresnel propagation, and assigns the ten output classes to ten detector regions. The film is represented as one ideal phase screen during classifier training; exposure/readout imperfections enter only through future measurements. The exposure utility computes the LETO pattern for a target field at the film and does not simulate the film's index modulation.

If your measured film active thickness, refractive index, illumination aperture, relay distances, laser wavelength, or recording angles differ, edit `config.json` or pass the relevant command options before interpreting a result. Do not compare a small training run's accuracy directly with the paper's five-layer result.

## Code map

| File | Purpose |
|---|---|
| `optics.py` | Differentiable Fresnel propagation, phase screen, input encoding, detector readout |
| `run.py` | Dataset split, TensorFlow training, held-out accuracy and saved run artifacts |
| `capacity.py` | Nominal phase-cell budget plus one-grating angular selectivity screen |
| `multiplex.py` | Independent classifier and candidate LETO patterns for angular channels |
| `exposure.py` | LETO phase-only inverse design for one trained screen |
| `app.py` | Local browser interface to inspect runs and change the cell-count experiment |
| `config.json` | Editable starting geometry and numerical settings |

## References

- X. Lin et al., “All-optical machine learning using diffractive deep neural networks,” *Science* 361, 1004–1008 (2018), [doi:10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084).
- Covestro, [Bayfol HX200 product data](https://solutions.covestro.com/en/products/bayfol/bayfol-hx200_000000000086212099).

## License

Project code is MIT-licensed. MNIST is downloaded from the TensorFlow dataset service when needed and is not bundled here.
