"""Candidate Fourier-plane recording field, not a calibrated film exposure."""
import argparse,json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image


def fftc(u):
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(u,axes=(-2,-1))),axes=(-2,-1))


def ifftc(u):
    return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(u,axes=(-2,-1))),axes=(-2,-1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--focal-mm',type=float,default=200);p.add_argument('--lut',help='256 measured phases in radians for current device configuration')
    args=p.parse_args()
    if args.steps<1 or args.focal_mm<=0:p.error('Positive steps and focal length required')
    folder=Path(args.run);c=json.loads((folder/'config.json').read_text())
    if c['layers']!=1: p.error('Single-film export requires a one-layer run')
    cell=np.load(folder/'phase_cells.npy')[0]
    h,w=1080,1920;pitch=6.4e-6;side=pitch*np.sqrt(.93)
    lf=c['wavelength_nm']*1e-9*args.focal_mm*1e-3
    x=np.fft.fftshift(np.fft.fftfreq(w,pitch))*lf
    y=np.fft.fftshift(np.fft.fftfreq(h,pitch))*lf
    aperture=c['aperture_mm']*1e-3;k=c['cells']
    ix=np.clip(((x/aperture+.5)*k).astype(int),0,k-1)
    iy=np.clip(((y/aperture+.5)*k).astype(int),0,k-1)
    roi=(np.abs(y[:,None])<aperture/2)&(np.abs(x[None,:])<aperture/2)
    if aperture/k < 2*max(x[1]-x[0],y[1]-y[0]):
        p.error('Film cells under-resolved at this Fourier sampling; increase sampling before export')
    desired=np.exp(1j*cell[iy[:,None],ix[None,:]])*roi
    # Pixel-integrated thin lens, SLM one focal length before lens.
    t=(-1j*side**2/lf)*np.sinc(side*y[:,None]/lf)*np.sinc(side*x[None,:]/lf)
    t=tf.constant(t,tf.complex64)
    back=ifftc(tf.constant(desired,tf.complex64)/t)
    scale=.35/float(tf.reduce_max(tf.abs(back)))
    target=tf.constant(desired*scale,tf.complex64)
    phase=tf.Variable(tf.math.angle(back));opt=tf.keras.optimizers.Adam(.05)
    # Fit full computed plane, so light outside desired field is penalized too.
    def forward(ph):return fftc(tf.exp(tf.complex(tf.zeros_like(ph),ph)))*t
    def loss(ph):return tf.reduce_sum(tf.abs(forward(ph)-target)**2)/tf.reduce_sum(tf.abs(target)**2)
    best=float(loss(phase));bestphase=phase.numpy();history=[best]
    @tf.function
    def step():
        with tf.GradientTape() as tape:l=loss(phase)
        opt.apply_gradients([(tape.gradient(l,phase),phase)])
        return loss(phase)
    for i in range(args.steps):
        value=float(step());history.append(value)
        if value<best:best=value;bestphase=phase.numpy()
        if (i+1)%10==0:print(f'exposure fit {i+1}/{args.steps}: {value:.5g}',flush=True)
    wrapped=np.mod(bestphase,2*np.pi)
    if args.lut:
        lut=np.load(args.lut)
        if lut.shape!=(256,) or not np.isfinite(lut).all():raise ValueError('LUT must contain 256 finite phases')
        # Bounded memory: search one grayscale code at a time.
        error=np.full((h,w),np.inf);gray=np.zeros((h,w),np.uint8)
        for code,value in enumerate(lut):
            e=np.abs(np.angle(np.exp(1j*(wrapped-value))));mask=e<error
            gray[mask]=code;error[mask]=e[mask]
        actual=lut[gray]
    else:
        gray=(np.rint(wrapped*256/(2*np.pi)).astype(int)%256).astype(np.uint8)
        actual=gray.astype(np.float32)*2*np.pi/256
    predicted=forward(tf.constant(actual,tf.float32)).numpy()
    a=target.numpy()[roi];b=predicted[roi]
    report={'scope':'Candidate object-arm field only. Separate coherent reference required. No material dose or readout model.',
      'calibration':'Measured supplied LUT' if args.lut else 'IDEAL LUT; uncalibrated',
      'wavelength_nm':c['wavelength_nm'],'focal_mm':args.focal_mm,'slm_lens_mm':args.focal_mm,
      'target_scale':scale,'roi_complex_mse':float(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2)),
      'roi_fidelity':float(np.abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real)),
      'full_plane_loss':history,'sampling_note':'FFT samples only; no between-sample validation. Validate before hardware use.'}
    out=folder/'exposure';out.mkdir(exist_ok=True)
    np.save(out/'phase_radians.npy',wrapped)
    np.savez_compressed(out/'object_field.npz',target=target.numpy(),predicted=predicted,x_m=x,y_m=y,roi=roi)
    Image.fromarray(gray).save(out/('SLM_LUT.png' if args.lut else 'SLM_IDEAL_LUT.png'))
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='full_plane_loss'},indent=2))

if __name__=='__main__':main()
