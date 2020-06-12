import numpy as np
from scipy import ndimage

def center_of_mass(histogram):
    """ center_of_mass: use scipy
    return ndimage.measurements.center_of_mass(input=histogram) """
    return ndimage.measurements.center_of_mass(input=histogram)

#a faster way of CenterOfMass (compared to recursive version and scipy com ,but has integer overflow sometimes)
def histogram_center(histogram, axis=None, verbose=False, convert_to_datatype=None):
    """histogram_center: calculate Center of mass of numpy array of histogram, by matrix multiplication using array coordinates. 
    Center of mass is calculated for dimensions specified with axis sequence. The function is vectorized.
    Make sure that the array datatype has no overflow issues with np.sum(arr). 
    histogram, axis=None, verbose=False, convert_to_datatype=None
    return: center_of_mass"""
    
    #default, calculate for all axes
    if axis is None:
        axis = tuple(np.arange(len(histogram.shape)))
        
    histogram = np.array(histogram, dtype=convert_to_datatype) 
        
    #sum removes axes
    counts = np.sum(histogram, axis=axis)
    
    #shape and storage for the results
    com_shape = [len(axis)]
    for k in range(np.ndim(histogram)):
        if k not in axis:
            com_shape.append(histogram.shape[k])
    vector_sum = np.zeros(shape=tuple(com_shape))
    
    for k,ax in enumerate(axis): 
        position_on_axis = np.arange(histogram.shape[ax])
        # add 1s to shape for broadcasting in numpy multiplication
        position_shape = np.ones(np.ndim(histogram), dtype=np.int64)
        # align axis position for broadcasting
        position_shape[ax] = histogram.shape[ax]
        # calculate array mass weighted sum of positions on axis
        vector_sum[k] = np.sum(histogram*np.reshape(position_on_axis, tuple(position_shape)), axis=axis)
        
        if verbose:
            print("axis ", k," = ", ax)
            print("position shape\n", position_shape)
            print("position_on_axis \n", position_on_axis)
    
    if verbose:
        print("counts shape\n",counts.shape) 
        print("com shape\n", vector_sum.shape)
        print("array shape\n", histogram.shape)
    
    #normalize
    return vector_sum / np.maximum(1.0, counts)

def listmode_center(listmode_data, axis=1, masses=None):
    """ listmode_center: calculate Center of mass of 2D numpy array with dimension coordinates. Points listed along axis.
        input: listmode, axis=1, masses=None
        if no masses provided, mass of 1 for all points is assumed
        return center_of_mass"""
    if masses is None:
        center_of_mass = np.mean(listmode_data, axis=axis)
    else:
        center_of_mass = np.sum(masses*listmode_data, axis=axis)
        center_of_mass /= np.sum(masses)
    
    return center_of_mass

def get_center_trace(listmode_data, split_indices, masses=None):
    """ get_center_trace: split the listmode data and get the center of mass, optionally weighted by masses
    input: listmode_data, split_indices, masses=None
    return trace"""
    ndims = len(listmode_data)
    pieces = np.split(listmode_data, split_indices, axis=1)

    if masses is not None:
        mass_pieces = np.split(masses, split_indices, axis=0)
    trace = np.empty((ndims, len(pieces)), dtype=listmode_data.dtype)
    for k, p in enumerate(pieces):
        if masses is None:
            slice_mass = None
        else:
            slice_mass = mass_pieces[k]
        trace[:, k] = listmode_center(p, axis=1, masses=slice_mass)

    return trace