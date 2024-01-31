import os, re
import numpy as np
from PIL import Image
import pandas as pd
import argparse
from tqdm import tqdm

def get_files_path_recursively(path, *args, suffix=''):
    """Retrieve specific files path recursively from directory.

    Retrieve the path of all files with one of the given extension names,
    in the given directory and all its subdirectories, recursively.
    The extension names should be given as a list of strings. The search for
    extension names is case sensitive.

    Args:
        path (str): root directory from which to search for files recursively
        *args: list of file extensions to be considered

    Returns:
        list(str): list of paths of every files in the directory and all its
        subdirectories
    """
    exts = list(args)
    for ext_i, ext in enumerate(exts):
        exts[ext_i] = ext[1:] if ext[0] == '.' else ext
    ext_list = "|".join(exts)
    result = [os.path.join(dp, f)
              for dp, dn, filenames in os.walk(path)
              for f in filenames
              if re.search(rf"^.*({suffix})\.({ext_list})$", f)]
    return result

def standardize(root_path:str='sample_data/SatelliteImages/',
                ext:str=['jpeg', 'jpg'],
                output:str='root_path'):
    """Perform standardization over images.
    
    Returns and stores the mean and standard deviation of an image
    dataset organized inside a root directory for computation
    purposes like deep learning.

    Args:
        root_path (str): root dir. containing the images.
                         Defaults to './sample_data/SatelliteImages/'.
        ext (str, optional): the images extensions to consider.
                             Defaults to 'jpeg'.
        output (str, optional): output path where to save the csv containing
                                the mean and std of the dataset.
                                If None: doesn't output anything.
                                Defaults to root_path.

    Returns:
        _type_: _description_
    """
    fps = get_files_path_recursively(root_path, *ext)
    imgs = []
    stats = {'mean':[], 'std':[]}
    for fp in fps:
        img = np.array(Image.open(fp, mode='r'))
        if len(img.shape) == 3:
            img = np.transpose(img, (2,0,1))
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=0)
    stats['mean'].append(np.nanmean(imgs))
    stats['std'].append(np.nanstd(imgs))
    if output:
        output = os.path.join(root_path, 'standardize_stats.csv') if output=='root_path' else output
        df = pd.DataFrame(stats)
        df.to_csv(output, index=False, sep=';')
    return stats['mean'][0], stats['std'][0]

def per_channel_mean_std(root_path, output, ext=['jpg', 'jpeg']):
    files = get_files_path_recursively(root_path, *ext)
    dims = len(np.array(Image.open(files[0], mode='r')).shape)

    imgs = []
    for f in files:
        img = np.array(Image.open(f, mode='r'))
        if dims == 3:
            img = np.transpose(img, (2,0,1))
        elif  dims == 2:
            img = np.expand_dims(img, axis=0)
        imgs.append(img)
    imgs = np.array(imgs)

    stats = pd.DataFrame({
        'mean': np.nanmean(imgs,axis=(0,2,3)),
        'std': np.nanstd(imgs,axis=(0,2,3))
    })
    stats.to_csv(output, index=True, sep=';')

def per_channel_mean(root_path, output, ext=['jpg', 'jpeg']):
    files = get_files_path_recursively(root_path, *ext)
    img_shape = np.array(Image.open(files[0], mode='r')).shape 
    dims = len(img_shape)

    if dims == 3:
        sums = np.array([0.0] * img_shape[2]) 
    elif dims == 2:
        sums = np.array([0.0])
    n_terms = 0

    for f in tqdm(files):
        img = np.array(Image.open(f, mode='r'))
        img_sum = np.sum(img, axis=(0,1))
        sums += img_sum
        n_terms += img.shape[0] * img.shape[1]

    means = sums / n_terms
    means_df = pd.DataFrame({'mean': means})
    means_df.to_csv(output, index=False)

def per_channel_std(root_path, means_path, output, ext=['jpg', 'jpeg']):
    files = get_files_path_recursively(root_path, *ext)
    means = np.array( pd.read_csv(means_path)['mean'])
    img_shape = np.array(Image.open(files[0], mode='r')).shape 
    dims = len(img_shape)

    if dims == 3:
        sums = np.array([0.0] * img_shape[2]) 
    elif dims == 2:
        sums = np.array([0.0])
    n_terms = 0

    for f in tqdm(files):
        img = np.array(Image.open(f, mode='r'))
        sum_dev_mean = np.sum(np.power(img - means, 2), axis=(0,1))
        sums += sum_dev_mean
        n_terms += img.shape[0] * img.shape[1]
    
    std_devs = np.power(sums / n_terms, 0.5)
    std_devs_df = pd.DataFrame({'std_dev': std_devs})
    std_devs_df.to_csv(output, index=False)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--root_path',
                        nargs=1,
                        type=str,
                        help='Rooth path.')
    PARSER.add_argument('--out',
                        nargs=1,
                        type=str,
                        help='Output path.')
    PARSER.add_argument('--stat',
                        nargs=1,
                        type=str,
                        help='Statistic to compute ("mean" or "std")')
    PARSER.add_argument('--mean_path',
                        nargs=1,
                        type=str,
                        default=[''],
                        help='Path to csv with mean values for computation of standard deviation')

    ARGS = PARSER.parse_args()
    path = ARGS.root_path[0]
    out = ARGS.out[0]
    stat = ARGS.stat[0]
    mean_path = ARGS.mean_path[0]

    if stat == 'mean':
        per_channel_mean(path, out)
    elif stat == 'std':
        assert os.path.exists(mean_path)
        per_channel_std(path, mean_path, out)
    else:
        exit('Enter either "mean" of "std" as a statistic to compute')
       