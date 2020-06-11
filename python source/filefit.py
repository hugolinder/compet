import kex_headers
import kex_data
import os 
import shutil
import numpy as np

def prep_folder(dirname, copy_from):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for path in copy_from:
        shutil.copy(path, dirname, follow_symlinks=True)

def make_kex_sino_test_folder(destination, withsino=False):
    """ set up folder for testing.
    copies sinogram headerfiles from kex. 
    Returns corresponding paths in destination as dict:
    return spaths, hpaths, mpaths """
    sinos = kex_data.get_sino_paths()
    hdrs = kex_data.get_sino_hdrs()
    mhdrs = kex_data.get_sino_mainhdrs()
    for shm in zip(sinos,hdrs,mhdrs):
        filling = shm if withsino else shm[1:]
        prep_folder(destination, filling)

def refolder_path(folder, path):
    return folder + "/" + path.split("/")[-1]

def refolder_paths(folder, paths):
    return [ refolder_path(folder, p) for p in paths]

def get_data(path, shape=(109,200,200),dtype=np.float32):
    """ reads image from the .v interfile at path """
    count = np.prod(shape)
    with open(path, 'rb') as file:
        img = np.fromfile(file, dtype=dtype, count=count)
    return np.reshape(img, newshape=shape)

def get_v_data(path, shape=(109,200,200),dtype=np.float32):
    return get_data(path, shape, dtype) 

def get_path_selection(folder, filetype):
    """ return sequences of file paths in folder that ends with filetype """
    vpaths = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(filetype): 
                vpaths.append(folder+"/"+entry.name)
    return vpaths   