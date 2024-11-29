from utils import remamDGlob, train_testsplit
from skimage.transform import resize
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from skimage.io import imread, imsave


def prepare_dataset(in_dir, out_dir):
    impages = sorted(glob(f'{in_dir}/train/*_sat.jpg'))
    labels = sorted(glob(f'{in_dir}/train/*_mask.png'))

    subs = ['train', 'valid', 'test']

    for pr in subs:
        os.makedirs(f'{out_dir}/{pr}', exist_ok=True)


    alls = list(zip(impages, labels))
    for part in tqdm(subs):
        out = train_testsplit(files=alls, part=part)
        sub_ims = [a[0] for a in out]
        sub_lbls = [a[1] for a in out]

        for j in range(len(sub_ims)):
            img = imread(sub_ims[j])
            img = resize(img, output_shape=(256, 256, 3), preserve_range=True).astype(np.uint8)

            im_n = os.path.split(sub_ims[j])[1]

            lbl = imread(sub_lbls[j])
            lbl = (resize(lbl, output_shape=(256, 256, 3))*255).astype(np.uint8)
            lbl = remamDGlob(lbl, shape=256)
            lbl = np.argmax(lbl, axis=-1).astype(np.uint8)

            lb_n = os.path.split(sub_lbls[j])[1]


            imsave(fname=f'{out_dir}/{part}/{im_n}', arr=img, check_contrast=False)
            imsave(fname=f'{out_dir}/{part}/{lb_n}', arr=lbl, check_contrast=False)

if __name__ == "__main__":   
    if os.path.exists('/home/getch/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2'):
        path = '/home/getch/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2'
        out_dir = '/home/getch/ssl/GORILLA/DIGITAL_glob_dataset' # input('Please specificy directory to save preprocessed files: ')
        prepare_dataset(in_dir=path, out_dir=out_dir)
    else:
        import kagglehub
        path = kagglehub.dataset_download("balraj98/deepglobe-land-cover-classification-dataset")
        out_dir = '/home/getch/ssl/GORILLA/DIGITAL_glob_dataset' # input('Please specificy directory to save preprocessed files: ')
        prepare_dataset(in_dir=path, out_dir=out_dir)