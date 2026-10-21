"""Build independent MNIST phase-mask channels for an angular-multiplexing study.

Each channel is trained as a separate single-screen classifier and gets a LETO
object-arm candidate. This is a numerical channel set, not a Bayfol recorder.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from capacity import angular_channel_screen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default=None)
    p.add_argument('--channels', type=int, default=3)
    p.add_argument('--angle-spacing-deg', type=float, default=2.5)
    p.add_argument('--object-angle-deg', type=float, default=-48.16)
    p.add_argument('--reference-angle-deg', type=float, default=48.16)
    p.add_argument('--cells', type=int, default=None)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--train', type=int, default=2000)
    p.add_argument('--test', type=int, default=1000)
    p.add_argument('--exposure-steps', type=int, default=20)
    p.add_argument('--focal-mm', type=float, default=200)
    p.add_argument('--lut', default=None, help='Measured 256-value LETO phase LUT (.npy)')
    args = p.parse_args()
    if args.channels < 2 or args.angle_spacing_deg <= 0:
        p.error('Multiplexed mode requires at least 2 channels and positive angle spacing')
    if args.epochs < 1 or args.train < 1 or args.test < 1 or args.exposure_steps < 1:
        p.error('epochs, training/test counts, and exposure steps must be positive')

    config = json.loads(Path(args.config).read_text())
    out = Path(args.output or f"runs/multiplex-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=False)
    offsets = (np.arange(args.channels) - (args.channels - 1) / 2) * args.angle_spacing_deg
    records = []
    for index, offset in enumerate(offsets):
        channel_name = f'channel-{index + 1:02d}'
        channel_dir = out / channel_name
        ref_angle = args.reference_angle_deg + float(offset)
        train_cmd = [sys.executable, 'run.py', '--config', args.config,
                     '--output', str(channel_dir), '--epochs', str(args.epochs),
                     '--train', str(args.train), '--test', str(args.test),
                     '--seed', str(config['seed'] + index)]
        if args.cells is not None:
            train_cmd += ['--cells', str(args.cells)]
        subprocess.run(train_cmd, check=True)
        exposure_cmd = [sys.executable, 'exposure.py', '--run', str(channel_dir),
                        '--steps', str(args.exposure_steps), '--focal-mm', str(args.focal_mm)]
        if args.lut:
            exposure_cmd += ['--lut', args.lut]
        subprocess.run(exposure_cmd, check=True)
        description_path = channel_dir / 'exposure' / 'LETO_pattern_description.json'
        description = json.loads(description_path.read_text())
        description.update({
            'multiplex_channel': index + 1,
            'multiplex_channel_count': args.channels,
            'object_angle_external_deg': args.object_angle_deg,
            'reference_angle_external_deg': ref_angle,
            'channel_angle_spacing_deg': args.angle_spacing_deg,
            'channel_interpretation': 'Record this object field simultaneously with a coherent reference at the listed angle. This package does not calculate fringes or exposure dose.'
        })
        description_path.write_text(json.dumps(description, indent=2) + '\n')
        records.append({
            'channel': index + 1,
            'directory': channel_name,
            'reference_angle_external_deg': ref_angle,
            'leto_image': f'{channel_name}/exposure/{description["leto_drive_image"]}',
            'test_accuracy': json.loads((channel_dir / 'metrics.json').read_text())['test']['accuracy'],
        })

    screen = angular_channel_screen(
        args.channels, args.angle_spacing_deg, args.object_angle_deg,
        args.reference_angle_deg, config['wavelength_nm'] * 1e-9,
        config['film_thickness_um'] * 1e-6, config['film_index'])
    display_dir = out / 'LETO_FILES_TO_DISPLAY'
    display_dir.mkdir()
    for record in records:
        channel_dir = out / record['directory'] / 'exposure'
        image_name = Path(record['leto_image']).name
        stem = f"LETO_CHANNEL_{record['channel']:02d}_REF_{record['reference_angle_external_deg']:.2f}deg"
        shutil.copy2(channel_dir / image_name, display_dir / f'{stem}.png')
        shutil.copy2(channel_dir / 'drive_phase_radians.npy', display_dir / f'{stem}_phase_radians.npy')
    readme = (
        'LETO FILES TO DISPLAY\n\n'
        'The .png files are 1920x1080 grayscale SLM drive frames, one per simulated channel. '
        'Each channel is assigned the reference angle in its filename. The .npy files are numeric '
        'phase arrays for inspection and are not display images.\n\n'
        'These are uncalibrated candidate patterns using the ideal gray-to-phase LUT. They are '
        'not exposure recipes: recording also requires the coherent reference beam, measured '
        'LETO calibration, and validated optical geometry.\n'
    )
    (display_dir / 'README.txt').write_text(readme)
    manifest = {
        'scope': 'Numerical angular-channel example: separately trained one-screen classifiers, each with one LETO object-arm phase candidate.',
        'physical_layer_count': 1,
        'bayfol_recording_volume_count': 1,
        'channel_count': args.channels,
        'same_film_patch_reused_for_each_recording': True,
        'display_files_folder': 'LETO_FILES_TO_DISPLAY',
        'reference_angle_spacing_deg': args.angle_spacing_deg,
        'channels': records,
        'weak_grating_angular_screen': screen,
        'limitations': [
            'Patterns are object-arm phase drives; separate coherent reference beams and sequential exposures are required.',
            'No fringe pattern, Bayfol dose, recording competition, shrinkage, or reconstructed classifier is simulated.',
            'Reference angles and LETO gray-to-phase calibration must be measured for hardware use.'
        ]
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'channels': records}, indent=2))


if __name__ == '__main__':
    main()
