"""Read-only result inspector plus explicit commands; never drives hardware."""
import json
import io
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from capacity import phase_budget
st.set_page_config(page_title='Bayfol MNIST lab',layout='wide')
st.title('Bayfol MNIST lab')

def show_leto_phase(image_path, phase_path, name):
    drive=np.asarray(Image.open(image_path))
    phase=np.load(phase_path) if phase_path.exists() else drive.astype(np.float32)*(2*np.pi/256)
    phase_buffer=io.BytesIO(); np.save(phase_buffer,phase)
    phase_bytes=phase_path.read_bytes() if phase_path.exists() else phase_buffer.getvalue()
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    im=ax.imshow(phase,cmap='twilight',vmin=0,vmax=2*np.pi,interpolation='nearest')
    ax.set_title(f'{name}: phase applied at each LETO pixel')
    ax.set_xlabel('LETO pixel column'); ax.set_ylabel('LETO pixel row')
    fig.colorbar(im,ax=ax,label='Phase (radians)')
    st.pyplot(fig); plt.close(fig)
    st.caption('The grayscale PNG is the drive-code image. This plot shows the corresponding phase at each SLM pixel. Under the ideal LUT, gray code g means phase = g × 2π/256 radians; your measured LETO calibration LUT replaces this mapping.')
    left,right=st.columns(2)
    left.download_button('Download exact phase array (.npy)',phase_bytes,file_name=f'{name.lower().replace(" ","-")}-phase-radians.npy',mime='application/octet-stream',key=f'{name}-phase-download')
    right.download_button('Download LETO drive image (.png)',image_path.read_bytes(),file_name=f'{name.lower().replace(" ","-")}-{image_path.name}',mime='image/png',key=f'{name}-image-download')
    row=st.slider(f'{name}: inspect LETO pixel row',0,phase.shape[0]-1,phase.shape[0]//2,key=f'{name}-row')
    col=st.slider(f'{name}: inspect LETO pixel column',0,phase.shape[1]-1,phase.shape[1]//2,key=f'{name}-col')
    st.write(f'Pixel (row {row}, column {col}): drive code **{int(drive[row,col])}**, phase **{phase[row,col]:.5f} rad** ({np.degrees(phase[row,col]):.2f}°).')
st.write('MNIST digit → coherent field → trained phase mask → propagation → ten detectors')
st.info('One physical 16 µm Bayfol film and one diffractive layer. Choose a single phase mask or an optional numerical angular-channel set with separate trained masks and LETO candidate patterns. Angular channels share the same film volume and are assigned different reference/readout angles. The film recording and readout are not physically simulated.')
latest_multiplex=sorted(Path('runs').glob('multiplex-*/manifest.json'),reverse=True)
public_multiplex=Path('demo_artifacts/multiplex-example/manifest.json')
if public_multiplex.exists(): latest_multiplex.append(public_multiplex)
if latest_multiplex:
    latest_root=latest_multiplex[0].parent
    latest_manifest=json.loads(latest_multiplex[0].read_text())
    latest_files=latest_root/latest_manifest.get('display_files_folder','LETO_FILES_TO_DISPLAY')
    st.subheader('Your LETO display files')
    st.write('Use the **PNG frames** below as the image files to display on the LETO. The phase `.npy` files are numeric arrays for inspection, not SLM image files.')
    st.code(str(latest_files.resolve()))
    file_cols=st.columns(min(3,len(latest_manifest['channels'])))
    for j,item in enumerate(latest_manifest['channels']):
        fname=f"LETO_CHANNEL_{item['channel']:02d}_REF_{item['reference_angle_external_deg']:.2f}deg.png"
        image_file=latest_files/fname
        with file_cols[j%len(file_cols)]:
            st.write(f"**Channel {item['channel']}** · {item['reference_angle_external_deg']:.2f}°")
            if image_file.exists():
                st.download_button(f'Download LETO PNG — channel {item["channel"]}',image_file.read_bytes(),file_name=fname,mime='image/png',key=f'top-leto-png-{item["channel"]}')
            else:
                st.caption(f"Pattern file is missing: {image_file}")
    st.warning('These are the current ideal-LUT candidate frames. Confirm the measured LETO gray-to-phase calibration before displaying them as phase commands or exposing film.')
paths=sorted(Path('runs').glob('*/metrics.json'),reverse=True)
public_single=Path('demo_artifacts/single/metrics.json')
if public_single.exists(): paths.append(public_single)
if not paths:
    st.code('python run.py --epochs 1 --train 2000 --test 1000'); st.stop()
run=st.selectbox('Saved experiment',[str(p.parent) for p in paths]); p=Path(run)
c=json.loads((p/'config.json').read_text()); m=json.loads((p/'metrics.json').read_text())
a,b,d=st.columns(3)
a.metric('Held-out accuracy',f"{m['test']['accuracy']:.1%}");b.metric('Adjustable phase parameters',m['phase_parameters']);d.metric('Cell pitch at film',f"{m['cell_pitch_um']:.1f} µm")
st.image(str(p/'summary.png'))
st.write('The phase cells are adjustable screen values, not freely programmable connections between every input and output point.')
example_path=p/'examples.npz'
if example_path.exists():
    data=np.load(example_path); i=st.slider('Test example',0,len(data['labels'])-1,0)
    a,b=st.columns(2); a.image(data['images'][i],caption=f"MNIST digit {data['labels'][i]}",width=220)
    u=data['intensity'][i]; b.image(u/max(float(u.max()),1e-20),caption='Normalized detector-plane intensity',clamp=True,width=350)
    st.write('Detector label layout: top row 0–4, bottom row 5–9.')
else:
    st.caption('Digit examples are omitted from the public hosted demo; the saved classifier summary and LETO patterns are shown above.')
st.subheader('Change the phase budget')
k=st.select_slider('Cells along each axis',options=[4,8,16,32,64],value=16)
budget=phase_budget(c['aperture_mm'],int(k))
st.write(f"{budget['phase_values']:,} phase values ({budget['intensity_observable_phase_degrees_of_freedom']:,} intensity-observable degrees of freedom), with {budget['cell_pitch_um']:.1f} µm cells over the configured {c['aperture_mm']:.3f} mm aperture. This is the numerical mask budget, not a measured film capacity.")
st.caption(f"The {c['film_thickness_um']} µm thickness mainly sets volume-grating angular selectivity. It does not alone set how many lateral phase cells the film can resolve. Measure phase-map reconstruction and detector cross-talk to find that limit.")
st.subheader('Angular multiplexing: channels in one film')
channels=st.slider('Recorded angular channels (estimate)',2,9,3)
spacing=st.slider('Reference-angle spacing (degrees)',0.1,5.0,1.0,0.1)
angle_a,angle_b=st.columns(2)
object_angle=angle_a.slider('Object-beam angle for estimate (° from film normal)',-85.0,85.0,-48.16,0.1)
reference_angle=angle_b.slider('Center reference-beam angle for estimate (° from film normal)',-85.0,85.0,48.16,0.1)
from capacity import angular_channel_screen
film_index=c.get('film_index',1.5)
screen=angular_channel_screen(channels,spacing,object_angle,reference_angle,c['wavelength_nm']*1e-9,c['film_thickness_um']*1e-6,film_index)
st.write('The object angle is held fixed across channels. The reference angle is stepped around its center by the selected spacing; each recording is assigned its corresponding reference angle. On readout, set the reference/readout beam to that channel angle to select it. These settings affect the angle estimate and channel labels only: they do not steer your beams or change the LETO pixel pattern.')
st.caption(f"Current numerical geometry: object = {object_angle:.2f}°, center reference = {reference_angle:.2f}°, channel spacing = {spacing:.2f}°. Angles are external angles in air, signed from the film normal.")
st.warning('Geometry limitation: the current weak-grating calculation uses a simplified same-forward-direction wavevector model. It does not yet model a reflection hologram with counter-propagating longitudinal wavevectors. Since your Bayfol experiment is intended to be reflective, treat this angle matrix as a software demonstration only—not as predicted channel spacing for your setup.')
st.caption('Estimated normalized weak-grating response: rows are selected readout angles; columns are recorded holograms. Off-diagonal values indicate predicted angular leakage. This idealized calculation omits material saturation, exposure competition, shrinkage, strong-grating coupling, and measured setup geometry.')
st.dataframe(screen['off_channel_to_selected_channel_ratio'],hide_index=True)
st.code(f'python capacity.py --angular-channels {channels} --angle-spacing-deg {spacing:.1f} --object-angle-deg {object_angle:.2f} --reference-angle-deg {reference_angle:.2f}')
st.caption('Use measured recording/readout angles and a measured single-grating angular response before choosing channel spacing or channel count.')
st.code(f'python run.py --cells {k} --epochs 5 --train 55000 --test 10000')
st.subheader('LETO pattern simulation')
mode=st.radio('Simulation option',['Single phase mask','Angular-multiplexed channel set'],horizontal=True)
if mode=='Single phase mask':
    st.code(f'python exposure.py --run "{run}" --steps 100')
    exposure_dir=p/'exposure'
    image_path=exposure_dir/'SLM_LUT.png'
    if not image_path.exists(): image_path=exposure_dir/'SLM_IDEAL_LUT.png'
    if image_path.exists():
        em=json.loads((exposure_dir/'metrics.json').read_text())
        if 'leto_image_resolution_pixels' not in em:
            field=np.load(exposure_dir/'object_field.npz')
            drive=np.asarray(Image.open(image_path))
            phase=np.load(p/'phase_cells.npy')
            em.update({'leto_image_resolution_pixels':[int(drive.shape[1]),int(drive.shape[0])],
              'learned_phase_cells_xy':[int(phase.shape[-1]),int(phase.shape[-2])],
              'film_target_square_aperture_mm':[c['aperture_mm'],c['aperture_mm']],
              'film_target_sample_count_xy':[int(np.count_nonzero(np.any(field['roi'],axis=0))),int(np.count_nonzero(np.any(field['roi'],axis=1)))],
              'learned_phase_cell_pitch_um':c['aperture_mm']*1000/phase.shape[-1],
              'recording_unit':'One continuous square target patch in one film area; not a hogel array.',
              'gray_to_phase_mapping':'Ideal linear mapping: gray 0..255 -> phase 0..2pi; not calibrated for the current LETO.'})
        st.image(str(image_path),caption=f"LETO phase-drive image · {image_path.name} · {em['leto_image_resolution_pixels'][0]}×{em['leto_image_resolution_pixels'][1]} pixels")
        st.write('**See the phase applied at every LETO pixel:**')
        show_leto_phase(image_path,exposure_dir/'drive_phase_radians.npy','Single mask')
        st.write(f"**What it displays:** a phase-only LETO drive pattern optimized so the modeled lens produces the learned {em['learned_phase_cells_xy'][0]}×{em['learned_phase_cells_xy'][1]} phase mask in the film plane. The mask occupies a {em['film_target_square_aperture_mm'][0]:.4f}×{em['film_target_square_aperture_mm'][1]:.4f} mm square, represented by about {em['film_target_sample_count_xy'][0]}×{em['film_target_sample_count_xy'][1]} sampled film-plane points. The phase-cell pitch is {em['learned_phase_cell_pitch_um']:.1f} µm.")
        st.info(f"**Recording unit:** {em['recording_unit']} The LETO image specifies the object arm only. A separate coherent reference beam must overlap this target on the film to record fringes. This PNG does not include those fringes or predict Bayfol exposure/readout.")
        st.warning(f"**Drive calibration:** {em['gray_to_phase_mapping']}")
    else:
        st.caption('Run the command above to generate the one-channel LETO pattern.')
else:
    st.write('This option trains a separate single-screen MNIST classifier for each angular channel and creates a matching LETO object-arm pattern for each one. All channels are assigned to the same film patch, with different reference angles. This is a numerical multiplexed channel set, not a material recording/readout simulation.')
    st.code(f'python multiplex.py --channels {channels} --angle-spacing-deg {spacing:.1f} --object-angle-deg {object_angle:.2f} --reference-angle-deg {reference_angle:.2f} --epochs 3 --train 2000 --test 1000 --exposure-steps 20')
    multiplex_runs=sorted(Path('runs').glob('multiplex-*/manifest.json'),reverse=True)
    if public_multiplex.exists(): multiplex_runs.append(public_multiplex)
    if multiplex_runs:
        chosen=st.selectbox('Saved multiplexed simulation',[str(x.parent) for x in multiplex_runs])
        root=Path(chosen); manifest=json.loads((root/'manifest.json').read_text())
        st.write(f"{manifest['channel_count']} channels · {manifest['reference_angle_spacing_deg']}° spacing · one modeled recording volume")
        tabs=st.tabs([f"Channel {item['channel']} · {item['reference_angle_external_deg']:.2f}°" for item in manifest['channels']])
        for tab,item in zip(tabs,manifest['channels']):
            with tab:
                image=root/item['leto_image']
                st.image(str(image),caption=f"Object-arm phase drive for channel {item['channel']}; reference angle {item['reference_angle_external_deg']:.2f}°; test accuracy {item['test_accuracy']:.1%}")
                show_leto_phase(image,image.parent/'drive_phase_radians.npy',f"Channel {item['channel']}")
        st.dataframe(manifest['weak_grating_angular_screen']['off_channel_to_selected_channel_ratio'],hide_index=True)
        st.warning('To record the channels, display each channel pattern while overlapping the corresponding coherent reference angle on the same film patch. The files do not contain fringes, dose settings, or validated hardware calibration.')
    else:
        st.caption('No multiplexed pattern set exists yet. Run the command above in the project terminal; then refresh this page to inspect the channel images.')
with st.expander('Geometry and all metrics'): st.json(c); st.json(m)
