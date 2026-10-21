"""Differentiable scalar Fresnel baseline; not a volume photopolymer solver."""
import numpy as np
import tensorflow as tf


def transfer(n, pitch, wavelength, distance):
    # Unshifted FFT frequencies. Remove transfer frequencies whose phase
    # changes by > pi/sample (conservative axiswise Fresnel band limit).
    f = np.fft.fftfreq(n, pitch)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    h = np.exp(-1j*np.pi*wavelength*distance*(fx*fx+fy*fy))
    if distance:
        limit = n*pitch/(2*wavelength*abs(distance))
        h *= (np.abs(fx)<=limit)&(np.abs(fy)<=limit)
    return tf.constant(h, tf.complex64)


def propagate(u, h):
    return tf.signal.ifft2d(tf.signal.fft2d(u)*h)


def detectors(n, grid):
    masks=np.zeros((10,n,n),np.float32)
    start=(n-grid)//2
    side=max(2,grid//12)
    for label in range(10):
        row,col=divmod(label,5)
        y=start+int(grid*(.35+.30*row)); x=start+int(grid*(.15+.175*col))
        masks[label,y-side//2:y-side//2+side,x-side//2:x-side//2+side]=1
    return masks


class DiffractiveClassifier(tf.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        g,k=c['grid'],c['cells']
        if g<32 or g%k or c['padding']<2 or c['layers']<1:
            raise ValueError('grid >=32, cells must divide grid, padding >=2, layers >=1')
        if min(c['wavelength_nm'],c['aperture_mm'],c['distance_mm'],c['film_thickness_um'])<=0:
            raise ValueError('Physical dimensions must be positive')
        if c['aperture_mm']>min(c['sheet_width_mm'],c['sheet_height_mm']):
            raise ValueError('Aperture exceeds sheet')
        self.n=g*c['padding']; self.pad=(self.n-g)//2
        self.h=transfer(self.n,c['aperture_mm']*1e-3/g,c['wavelength_nm']*1e-9,c['distance_mm']*1e-3)
        self.masks=tf.constant(detectors(self.n,g))
        self.phase=tf.Variable(tf.random.stateless_uniform((c['layers'],k,k),[c['seed'],0],-.1,.1),name='phase')

    def expanded(self):
        repeat=self.c['grid']//self.c['cells']
        return tf.repeat(tf.repeat(self.phase,repeat,axis=1),repeat,axis=2)

    def input_field(self,images):
        g=self.c['grid']; size=g//2
        # Grayscale is field amplitude, not intensity. Normalize incident power.
        amp=tf.image.resize(tf.cast(images[...,None],tf.float32)/255.,(size,size),method='bilinear')[...,0]
        before=(self.n-size)//2
        amp=tf.pad(amp,[[0,0],[before,self.n-size-before],[before,self.n-size-before]])
        amp/=tf.sqrt(tf.reduce_sum(amp**2,axis=(-2,-1),keepdims=True)+1e-12)
        return tf.complex(amp,tf.zeros_like(amp))

    def __call__(self,images):
        u=self.input_field(images)
        for phase in tf.unstack(self.expanded()):
            u=propagate(u,self.h)
            t=tf.exp(tf.complex(tf.zeros_like(phase),phase))
            # Finite illuminated mask aperture: zero outside, not periodic tiles.
            t=tf.pad(t,[[self.pad,self.n-self.c['grid']-self.pad]]*2)
            u*=t
        u=propagate(u,self.h)
        intensity=tf.abs(u)**2
        energy=tf.einsum('byx,kyx->bk',intensity,self.masks)
        probabilities=(energy+1e-12)/(tf.reduce_sum(energy,axis=1,keepdims=True)+1e-11)
        return probabilities,energy,intensity
