import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from optics import DiffractiveClassifier,transfer,propagate


def config():
    return dict(json.loads(Path('config.json').read_text()),grid=32,cells=8)


def test_zero_distance_identity_and_power():
    rng=np.random.default_rng(1)
    u=tf.constant(rng.normal(size=(2,32,32))+1j*rng.normal(size=(2,32,32)),tf.complex64)
    v=propagate(u,transfer(32,6.4e-6,532e-9,0))
    np.testing.assert_allclose(v,u,atol=1e-6)
    np.testing.assert_allclose(tf.reduce_sum(tf.abs(v)**2),tf.reduce_sum(tf.abs(u)**2),rtol=1e-6)


def test_global_phase_invariance_and_nonzero_gradient():
    m=DiffractiveClassifier(config());x=np.random.default_rng(2).integers(0,256,(2,28,28)).astype(np.uint8)
    p,e,_=m(x)
    m.phase.assign_add(tf.ones_like(m.phase)*.7)
    q,f,_=m(x)
    np.testing.assert_allclose(e,f,rtol=2e-5,atol=1e-7)
    np.testing.assert_allclose(tf.reduce_sum(q,1),1,atol=1e-6)
    assert float(tf.reduce_max(tf.reduce_sum(f,1)))<=1.00001
    with tf.GradientTape() as tape:
        prob,_,_=m(x);loss=-tf.math.log(prob[0,0])
    grad=tape.gradient(loss,m.phase).numpy()
    assert np.isfinite(grad).all() and np.linalg.norm(grad)>1e-5
    # Finite difference verifies differentiation through complex propagation.
    original=m.phase.numpy();idx=np.unravel_index(np.argmax(np.abs(grad)),grad.shape);eps=.002
    plus=original.copy();minus=original.copy();plus[idx]+=eps;minus[idx]-=eps
    m.phase.assign(plus);lp=float(-tf.math.log(m(x)[0][0,0]))
    m.phase.assign(minus);lm=float(-tf.math.log(m(x)[0][0,0]))
    np.testing.assert_allclose(grad[idx],(lp-lm)/(2*eps),rtol=.03,atol=1e-4)


def test_detectors_disjoint_and_phase_budget():
    m=DiffractiveClassifier(config())
    assert np.max(np.sum(m.masks.numpy(),axis=0))==1
    assert np.prod(m.phase.shape)==64
    assert np.all(np.sum(m.masks.numpy(),axis=(1,2))>0)
