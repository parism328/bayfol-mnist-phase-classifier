"""Read-only result inspector plus explicit commands; never drives hardware."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
from capacity import phase_budget
st.set_page_config(page_title='Bayfol MNIST lab',layout='wide')
st.title('Bayfol MNIST lab')
st.write('MNIST digit → coherent field → trained phase mask → propagation → ten detectors')
st.info('One numerical phase-screen baseline for one 16 µm Bayfol recording. No multiplexing is implemented and the film’s independent weight capacity has not been measured.')
paths=sorted(Path('runs').glob('*/metrics.json'),reverse=True)
if not paths:
    st.code('python run.py --epochs 1 --train 2000 --test 1000'); st.stop()
run=st.selectbox('Saved experiment',[str(p.parent) for p in paths]); p=Path(run)
c=json.loads((p/'config.json').read_text()); m=json.loads((p/'metrics.json').read_text())
a,b,d=st.columns(3)
a.metric('Held-out accuracy',f"{m['test']['accuracy']:.1%}");b.metric('Adjustable phase parameters',m['phase_parameters']);d.metric('Cell pitch at film',f"{m['cell_pitch_um']:.1f} µm")
st.image(str(p/'summary.png'))
st.write('The phase cells are adjustable screen values, not freely programmable connections between every input and output point.')
data=np.load(p/'examples.npz'); i=st.slider('Test example',0,len(data['labels'])-1,0)
a,b=st.columns(2); a.image(data['images'][i],caption=f"MNIST digit {data['labels'][i]}",width=220)
u=data['intensity'][i]; b.image(u/max(float(u.max()),1e-20),caption='Normalized detector-plane intensity',clamp=True,width=350)
st.write('Detector label layout: top row 0–4, bottom row 5–9.')
st.subheader('Change the phase budget')
k=st.select_slider('Cells along each axis',options=[4,8,16,32,64],value=16)
budget=phase_budget(c['aperture_mm'],int(k))
st.write(f"{budget['phase_values']:,} phase values ({budget['intensity_observable_phase_degrees_of_freedom']:,} intensity-observable degrees of freedom), with {budget['cell_pitch_um']:.1f} µm cells over the configured {c['aperture_mm']:.3f} mm aperture. This is the numerical mask budget, not a measured film capacity.")
st.caption(f"The {c['film_thickness_um']} µm thickness mainly sets volume-grating angular selectivity. It does not alone set how many lateral phase cells the film can resolve. Measure phase-map reconstruction and detector cross-talk to find that limit.")
st.code(f'python run.py --cells {k} --epochs 5 --train 55000 --test 10000')
st.subheader('Candidate LETO exposure image')
st.code(f'python exposure.py --run "{run}" --steps 100')
st.caption('This calculates an object-arm field for a separate reference beam. It does not predict the recorded classifier or an exposure dose.')
with st.expander('Geometry and all metrics'): st.json(c); st.json(m)
