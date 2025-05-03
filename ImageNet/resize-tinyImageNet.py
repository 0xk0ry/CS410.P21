from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
cpus = multiprocessing.cpu_count()
cpus = min(48,cpus)

# Set PATH to Tiny ImageNet root
PATH = Path('archive/tiny-imagenet-200')
DEST = Path('archive/tiny-imagenet-200-resized')
szs = (160, 352)

def resize_img(p, im, fn, sz):
    w, h = im.size
    ratio = min(h / sz, w / sz)
    im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)
    # Save to DEST, preserving subfolder structure after PATH
    new_fn = DEST / str(sz) / fn.relative_to(PATH)
    new_fn.parent.mkdir(parents=True, exist_ok=True)
    im.convert('RGB').save(new_fn)

def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs:
        resize_img(p, im, fn, sz)

def resize_imgs(files):
    with ProcessPoolExecutor(cpus) as e:
        e.map(partial(resizes, None), files)

# Collect train image files (train/<class>/images/*.JPEG)
train_files = list((PATH / 'train').glob('*/*/*.JPEG'))
# Collect val image files (val/images/*.JPEG)
val_files = list((PATH / 'val' / 'images').glob('*.JPEG'))

if __name__ == "__main__":
    for sz in szs:
        ssz = str(sz)
        (DEST / ssz / 'train').mkdir(parents=True, exist_ok=True)
        (DEST / ssz / 'val').mkdir(parents=True, exist_ok=True)

    resize_imgs(train_files)
    resize_imgs(val_files)