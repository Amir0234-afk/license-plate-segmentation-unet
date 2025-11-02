import os, json, argparse
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras

from .model_unet import build_unet
from .data_utils import load_image, load_mask
from .metrics import dice_coef

def set_seeds(seed=42):
    import random, numpy as _np, os as _os
    _os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed); _np.random.seed(seed); tf.random.set_seed(seed)
    _os.environ['TF_DETERMINISTIC_OPS']='1'

def data_pairs(img_dir, mask_dir):
    images = sorted(glob(os.path.join(img_dir, '*')))
    masks = []
    for p in images:
        base = os.path.splitext(os.path.basename(p))[0]
        found = None
        for ext in ('.png','.jpg','.jpeg'):
            cand = os.path.join(mask_dir, base + ext)
            if os.path.exists(cand): found = cand; break
        masks.append(found)
    return [(i,m) for i,m in zip(images,masks) if m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--outdir', default='models')
    ap.add_argument('--img-size', type=int, nargs=2, default=[256,256])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    set_seeds(42)
    os.makedirs(args.outdir, exist_ok=True)

    pairs = data_pairs(args.images, args.masks)
    if not pairs:
        raise SystemExit('No image/mask pairs found.')

    n = len(pairs); split = int(0.8*n)
    train_pairs = pairs[:split]; val_pairs = pairs[split:]

    def gen(data_pairs):
        H,W = args.img_size
        for x_path, m_path in data_pairs:
            x = load_image(x_path, (H,W)).astype(np.float32)/255.0
            y = load_mask(m_path, (H,W), num_classes=3)
            yield x, y

    tr = tf.data.Dataset.from_generator(lambda: gen(train_pairs),
        output_signature=(
            tf.TensorSpec(shape=(args.img_size[0], args.img_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(args.img_size[0], args.img_size[1], 3), dtype=tf.float32),
        )).shuffle(512).batch(args.batch).prefetch(tf.data.AUTOTUNE)

    va = tf.data.Dataset.from_generator(lambda: gen(val_pairs),
        output_signature=(
            tf.TensorSpec(shape=(args.img_size[0], args.img_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(args.img_size[0], args.img_size[1], 3), dtype=tf.float32),
        )).batch(args.batch).prefetch(tf.data.AUTOTUNE)

    model = build_unet(img_size=tuple(args.img_size), num_classes=3)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', dice_coef(3)])

    cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(args.outdir,'unet.keras'),
                                        monitor='val_loss', save_best_only=True)
    ]

    hist = model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=cb)

    metrics = {k: float(v[-1]) for k,v in hist.history.items() if isinstance(v, list) and v}
    with open(os.path.join(args.outdir,'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outdir,'class_map.json'),'w') as f:
        json.dump({'0':'outside','1':'border','2':'inside'}, f, indent=2)

    print('Saved model and metrics to', args.outdir)

if __name__ == '__main__':
    main()
