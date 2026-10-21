"""Train an MNIST optical classifier and save inspectable results."""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import tensorflow as tf
from optics import DiffractiveClassifier


def evaluate(model,x,y,batch):
    predictions=[]; captured=[]
    for i in range(0,len(x),batch):
        prob,energy,_=model(x[i:i+batch])
        predictions.extend(tf.argmax(prob,1).numpy().tolist())
        captured.extend(tf.reduce_sum(energy,1).numpy().tolist())
    predictions=np.array(predictions)
    cm=np.zeros((10,10),int); np.add.at(cm,(y,predictions),1)
    return {'accuracy':float(np.mean(predictions==y)), 'mean_detector_power_fraction':float(np.mean(captured)),
            'count':len(y),'confusion_matrix':cm.tolist()}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--config',default='config.json'); a.add_argument('--output',default=None)
    a.add_argument('--epochs',type=int,default=5); a.add_argument('--train',type=int,default=55000)
    a.add_argument('--test',type=int,default=10000); a.add_argument('--batch',type=int,default=32)
    a.add_argument('--cells',type=int); a.add_argument('--layers',type=int); a.add_argument('--padding',type=int)
    args=a.parse_args()
    if not (1<=args.train<=55000 and 1<=args.test<=10000 and args.epochs>0 and args.batch>0):
        a.error('train 1..55000; test 1..10000; epochs and batch positive')
    c=json.loads(Path(args.config).read_text())
    for key in ['cells','layers','padding']:
        if getattr(args,key) is not None: c[key]=getattr(args,key)
    tf.keras.utils.set_random_seed(c['seed'])
    model=DiffractiveClassifier(c); optimizer=tf.keras.optimizers.Adam(.03)
    (x,y),(xt,yt)=tf.keras.datasets.mnist.load_data()
    rng=np.random.default_rng(c['seed']); order=rng.permutation(60000)
    train_idx=order[:55000][:args.train]; val_idx=order[55000:]
    test_idx=rng.permutation(10000)[:args.test]
    xv,yv=x[val_idx],y[val_idx]; x,y=x[train_idx],y[train_idx]; xt,yt=xt[test_idx],yt[test_idx]
    out=Path(args.output or ('runs/'+time.strftime('%Y%m%d-%H%M%S')))
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    @tf.function
    def step(images,labels):
        with tf.GradientTape() as tape:
            prob,energy,_=model(images)
            ce=tf.keras.losses.sparse_categorical_crossentropy(labels,prob)
            # Small power term discourages correct answers with negligible light.
            loss=tf.reduce_mean(ce)-.02*tf.reduce_mean(tf.math.log(tf.reduce_sum(energy,1)+1e-9))
        grad=tape.gradient(loss,model.phase)
        tf.debugging.assert_all_finite(grad,'Nonfinite phase gradient')
        optimizer.apply_gradients([(grad,model.phase)])
        return loss
    history=[]; best=-1.; initial=evaluate(model,xt,yt,args.batch)
    for epoch in range(args.epochs):
        indices=rng.permutation(len(x)); losses=[]
        for i in range(0,len(x),args.batch):
            j=indices[i:i+args.batch]; losses.append(float(step(x[j],y[j])))
        val=evaluate(model,xv,yv,args.batch)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_accuracy':val['accuracy']}
        history.append(row); print(json.dumps(row),flush=True)
        if val['accuracy']>best:
            best=val['accuracy']; np.save(out/'phase_cells.npy',model.phase.numpy())
    model.phase.assign(np.load(out/'phase_cells.npy'))
    result=evaluate(model,xt,yt,args.batch)
    # Reevaluate identical masks/images on a wider FFT window: detect wraparound.
    wider=DiffractiveClassifier(dict(c,padding=c['padding']*2)); wider.phase.assign(model.phase)
    check=evaluate(wider,xt[:min(128,len(xt))],yt[:min(128,len(yt))],args.batch)
    base=evaluate(model,xt[:min(128,len(xt))],yt[:min(128,len(yt))],args.batch)
    np.save(out/'phase_masks_rad.npy',np.mod(model.expanded().numpy(),2*np.pi))
    _,_,intensity=model(xt[:10])
    np.savez_compressed(out/'examples.npz',images=xt[:10],labels=yt[:10],intensity=intensity.numpy(),detectors=model.masks.numpy())
    report={'scope':'Ideal phase-screen numerical baseline; no measured Bayfol response; no multiplexing',
        'tensorflow':tf.__version__,'python':platform.python_version(),'training_examples':len(x),
        'validation_examples':len(xv),'test':result,'initial_test':initial,'history':history,
        'phase_parameters':c['layers']*c['cells']**2,
        'intensity_observable_phase_dof_upper_bound':c['layers']*(c['cells']**2-1),
        'cell_pitch_um':1000*c['aperture_mm']/c['cells'],
        'film_independent_weights':'UNKNOWN: needs measured transfer/cross-talk and resolution',
        'padding_check':{'base':base,'double_padding':check,'samples':check['count']},
        'run_arguments':vars(args)}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    ax[0].imshow(model.expanded()[0],cmap='twilight'); ax[0].set_title('Learned phase, first layer [rad]')
    ax[1].plot([r['epoch'] for r in history],[r['validation_accuracy'] for r in history]); ax[1].set_title('Validation accuracy'); ax[1].set_ylim(0,1)
    ax[2].imshow(result['confusion_matrix']); ax[2].set_title('Held-out test confusion'); ax[2].set_xlabel('Predicted');ax[2].set_ylabel('True')
    fig.savefig(out/'summary.png',dpi=140);plt.close(fig)
    print(json.dumps({'output':str(out),'test_accuracy':result['accuracy'],'parameters':report['phase_parameters']},indent=2))

if __name__=='__main__': main()
