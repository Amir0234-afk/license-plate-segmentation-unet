import os, argparse, numpy as np
from glob import glob
from PIL import Image
from tensorflow import keras
from .data_utils import load_image

PALETTE = np.array([
    [0,0,0,0],        # outside transparent
    [255,255,255,255],# border white
    [0,0,0,255],      # inside black
], dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--img-size', type=int, nargs=2, default=[256,256])
    ap.add_argument('--overlay', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = keras.models.load_model(args.model, compile=False)

    paths = sorted(glob(os.path.join(args.images,'*')))
    for p in paths:
        x = load_image(p, tuple(args.img_size)).astype(np.float32)/255.0
        pred = model.predict(x[None,...], verbose=0)[0]
        cls = np.argmax(pred, axis=-1).astype(np.uint8)
        rgba = PALETTE[cls]
        base = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(rgba, mode='RGBA').save(os.path.join(args.out, f'{base}_mask.png'))
        if args.overlay:
            im = Image.open(p).convert('RGBA').resize((args.img_size[1], args.img_size[0]))
            over = Image.alpha_composite(im, Image.fromarray(rgba, 'RGBA'))
            over.save(os.path.join(args.out, f'{base}_overlay.png'))
    print('Wrote predictions to', args.out)

if __name__ == '__main__':
    main()
