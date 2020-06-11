import numpy as np

#Hugo code
import petlink
import representation
import kex_headers
import position

def join_gates(gates):
    return np.concatenate(gates, axis=-1)

def get_listmode(histogram):
    return np.array(np.nonzero(histogram))

def is_fov(listmode_data, shape):
    """ is_fov: checks if point bins are in fov (criterions: not less than 0 or too large for shape) 
    input: listmode_data  (coordinates), shape
    len(shape) == len(listmode_data) is assumed 
    return np.all(criterions, axis=0) """
    ndims = len(shape)

    criterions = np.zeros((2*ndims,np.size(listmode_data[0])), dtype=np.bool)
    for k in range(ndims):
        criterions[2*k] = listmode_data[k] > -1
        criterions[2*k+1] = listmode_data[k] < shape[k]
    return np.all(criterions, axis=0)

def is_within_sym_bound(listmode_data, bound):
    return (listmode < bound)*(-listmode < bound)

def get_histogram(listmode_data, shape, histo_at=None,weights=None,dtype=int):
    """ hisogram: histogramming, using numpy.add.at
    input: listmode_data, shape, histo=None,weights=None,dtype=None
    if weights is None:
        weights = 1
    return histogram """
    listmode_data = np.array(listmode_data, dtype=dtype)
    if weights is None:
        weights = 1
    if histo_at is None:
        histo_at = np.zeros(shape, dtype=dtype)
    np.add.at(histo_at, tuple(listmode_data), weights)
    return histo_at 

def get_sinogram(listmode_data, shape):
    events = petlink.get_events(listmode_data)
    histogram_bins = petlink.get_tof_lor_bins(events, shape)
    return get_histogram(histogram_bins, shape)

def get_prompt_sinogram(listmode_data, shape):
    events = petlink.get_events(listmode_data)
    prompts = petlink.get_prompts(events)
    histogram_bins = petlink.get_tof_lor_bins(prompts, shape)
    return get_histogram(histogram_bins, shape)

def get_tof_lor_bins(listmode_data, shape, events=None):
    if events is None:
        events = petlink.get_events(listmode_data)
    ba = petlink.get_bin_address(events)
    tup = np.unravel_index(ba, shape)
    return np.array(tup)

def get_fov_image_bins(listmode_data, **kwargs):
    """ get fov_image_bins: the pipeline from listmode to fov image bin
    input: petlink listmode_data, and keyvalue argumnts for representation.get_image_bins
    return dictionary with: is_event, events, tof_lor, is_prompt, prompts,  prompt_image_bins, is_image_fov, fov_image_bins  """
    is_event = petlink.is_event(listmode_data)
    events = listmode_data[is_event]
    tof_lor = get_tof_lor_bins(events, kex_headers.HISTOGRAM_SHAPE)

    is_prompt = petlink.is_prompt(events)
    #redundant
    prompts = events[is_prompt]

    prompt_image_bins = representation.get_img_bins2(tof_lor[:,is_prompt], kex_headers.SCAN_DICT)
    is_image_fov = is_fov(prompt_image_bins, kex_headers.IMAGE_SHAPE)
    fov_image_bins = prompt_image_bins[:,is_image_fov]

    dict = {"is_event" : is_event,
            "events" : events,
            "tof_lor": tof_lor,
            "is_prompt" :is_prompt,
            "prompts": prompts, 
            "prompt_image_bins": prompt_image_bins,
            "is_image_fov": is_image_fov,
            "fov_image_bins": fov_image_bins
            }
    return dict

def get_norms(tof_lor_bins, is_prompt, normfactors):
    """ get_norms: return a list of masses with correction factor per event including normalization and random correction (averaged over scan) 
    input: tof_lor_bins, is_prompt, normfactors
    return mass_list """
    is_random = [True, False]
    sinograms = np.empty((2,) + normfactors.shape)
    for is_r in is_random:
        LM = tof_lor_bins[1:, is_r != is_prompt]
        get_histogram(LM, shape=None, histo_at=sinograms[is_random.index(is_r)])
        if not is_r:
            prompt_LM = LM

    nonzero = sinograms[is_random.index(False)] > 0
    #the randoms are shared amongst the prompts. 
    ##so suppose there are 3 prompts and 2 randoms, then each prompt subtracts 2/3, for a total of -2
    sinograms[is_random.index(True)][nonzero] /= sinograms[is_random.index(False)][nonzero]
    #normalize
    sinograms[is_random.index(True)][nonzero] *= normfactors[nonzero]
    mass_sinogram = normfactors - sinograms[is_random.index(True)]
    #no negative mass allowed
    mass_sinogram = np.maximum(mass_sinogram, 0)
    return mass_sinogram[tuple(prompt_LM)]

def get_com_trace(listmode_data,timestep, normfactors=None):
    """ get_com_trace, the pipeline from raw listmode data to center of mass trace"""
    d = get_fov_image_bins(listmode_data)
    timetag_indices = petlink.get_timetag_indices(listmode_data)
    conditions = [d['is_event'], 
                  d['is_prompt'], 
                  d['is_image_fov']]
    for c in conditions:
        timetag_indices = transmit_indices(timetag_indices, c)
    split_indices = petlink.get_timeslice_indices(timetag_indices, timestep)
    if normfactors is None:
        masses=None
    else:
        #treat randoms as randoms, and prompts as prompts (including those outside fov)
        masses = get_norms(d['tof_lor'],d['is_prompt'],normfactors)
        masses = masses[d['is_image_fov']]
    return position.get_center_trace(d['fov_image_bins'], split_indices, masses=masses)

def transmit_indices(y_in_x, is_z_in_x):
    """ transmit indices: suppose a sequence x, with two subsequences y,z. Where would the indices of y-elements be in z?
    input: y_in_x, is_z_in_x
    return y_in_z """
    z_in_x = np.nonzero(is_z_in_x)[0]
    return np.searchsorted(z_in_x, y_in_x, side='left')

def transmit_belong_to(is_y_in_x, is_z_in_x):
    """ transmit_belong_to: suppose a sequence x, with two subsequences y,z. Where would the indices of y-elements be in z?
    input: is_y_in_x, is_z_in_x
    return y_in_z """
    y_in_x = np.nonzero(is_y_in_x)[0]
    return transmit_indices(y_in_x, is_z_in_x)

def convert_indices(is_y_in_x, is_z_in_x):
    """ converted_indices: return bounded indices of where y would be before elements in z
    input: is_y_in_x, is_z_in_x ( boolarrays)
    return z_in_y[z_in_y < ymax] """
    z_in_y = transmit_belong_to(is_y_in_x, is_z_in_x)
    ymax = np.sum(is_y_in_x)
    return z_in_y[z_in_y < ymax]

def convert_condition(is_y_in_x, is_z_in_x):
    """convert_conditions: get_boolean array indicating belonging with extra condition, where z is contained in y. 
    input: is_y_in_x, is_z_in_x
    example:
    X:      x x x x x 
    Y_of_X: _ y _ _ y 
    Z_of_X: z z _ z _  
    desired output: 
    Y:      y y
    Z_of_Y: z _ 
    return is z_in_y"""

    return np.all([is_y_in_x, is_z_in_x], axis=0)[is_z_in_x]

def is_a_in_c(is_a_in_b, is_b_in_c):
    """ is_a_in_c: 
    input: is_a_in_b, is_b_in_c (bool arrays) """
    a_of_c = np.zeros_like(is_b_in_c)
    a_of_c[is_b_in_c] = is_a_in_b
    return a_of_c

def expand_indicators(is_y_in_x, is_z_in_y):
    """ expand_indicators: 
    input: is_y_in_x, is_z_in_y
    return is_z_in_x """
    is_z_in_x = np.zeros_like(is_y_in_x)
    is_z_in_x[is_y_in_x] = is_z_in_y
    return is_z_in_x












    

    
