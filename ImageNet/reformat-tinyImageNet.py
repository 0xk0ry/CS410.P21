import os
import shutil
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../archive'))
ORIG_DIR = os.path.join(BASE_DIR, 'tiny-imagenet-200')
RESIZED_DIR = os.path.join(BASE_DIR, 'tiny-imagenet-200-resized')
REFORMAT_DIR = os.path.join(BASE_DIR, 'tiny-imagenet-200-reformat')

RESOLUTIONS = {
    'original': ORIG_DIR,
    '160': os.path.join(RESIZED_DIR, '160'),
    '352': os.path.join(RESIZED_DIR, '352'),
}

# Helper to copy images to ImageNet-like structure
def copy_class_train(src_class, dst_class):
    os.makedirs(dst_class, exist_ok=True)
    for img in glob(os.path.join(src_class, '*')):
        ext = os.path.splitext(img)[1].lower()
        if ext in ['.jpeg', '.jpg', '.png']:
            dst_img = os.path.join(dst_class, os.path.basename(img))
            shutil.copy(img, dst_img)

def copy_images(src_root, dst_root, split):
    src_split = os.path.join(src_root, split)
    dst_split = os.path.join(dst_root, split)
    if split == 'train':
        class_ids = os.listdir(os.path.join(src_split))
        with ThreadPoolExecutor() as executor:
            futures = []
            for class_id in class_ids:
                src_class = os.path.join(src_split, class_id, 'images') if os.path.exists(os.path.join(src_split, class_id, 'images')) else os.path.join(src_split, class_id)
                dst_class = os.path.join(dst_split, class_id)
                futures.append(executor.submit(copy_class_train, src_class, dst_class))
            for _ in as_completed(futures):
                pass
    elif split == 'val':
        val_img_dir = os.path.join(src_split, 'images') if os.path.exists(os.path.join(src_split, 'images')) else src_split
        val_annot = os.path.join(src_root, 'val', 'val_annotations.txt')
        if os.path.exists(val_annot):
            with open(val_annot) as f:
                lines = f.readlines()
            img2cls = {l.split('\t')[0]: l.split('\t')[1] for l in lines}
            def copy_val(img, class_id):
                src_img = os.path.join(val_img_dir, img)
                dst_class = os.path.join(dst_split, class_id)
                os.makedirs(dst_class, exist_ok=True)
                dst_img = os.path.join(dst_class, img)
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(copy_val, img, class_id) for img, class_id in img2cls.items()]
                for _ in as_completed(futures):
                    pass
        else:
            imgs = [img for img in glob(os.path.join(val_img_dir, '*')) if os.path.splitext(img)[1].lower() in ['.jpeg', '.jpg', '.png']]
            def copy_val_unknown(img):
                dst_class = os.path.join(dst_split, 'unknown')
                os.makedirs(dst_class, exist_ok=True)
                shutil.copy(img, os.path.join(dst_class, os.path.basename(img)))
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(copy_val_unknown, img) for img in imgs]
                for _ in as_completed(futures):
                    pass
    elif split == 'test':
        test_img_dir = os.path.join(src_split, 'images') if os.path.exists(os.path.join(src_split, 'images')) else src_split
        dst_class = os.path.join(dst_split, 'unknown')
        os.makedirs(dst_class, exist_ok=True)
        imgs = [img for img in glob(os.path.join(test_img_dir, '*')) if os.path.splitext(img)[1].lower() in ['.jpeg', '.jpg', '.png']]
        def copy_test(img):
            shutil.copy(img, os.path.join(dst_class, os.path.basename(img)))
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(copy_test, img) for img in imgs]
            for _ in as_completed(futures):
                pass

if __name__ == '__main__':
    for res, src_dir in RESOLUTIONS.items():
        dst_dir = os.path.join(REFORMAT_DIR, res)
        print(f'Processing {res}...')
        for split in ['train', 'val', 'test']:
            copy_images(src_dir, dst_dir, split)
        print(f'Done {res}.')
    print('All resolutions processed.')
