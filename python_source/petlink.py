import numpy as np 

def is_event(listmode):
    tag = (listmode >> 31) & 1
    return tag == 0
    
def get_events(listmode):
    return listmode[is_event(listmode)]

def is_timetag(listmode):
    """ is_timetag: must be a tag, and a timetag
    input: listmode data
    return bool array indicating time tags in listmode data"""
    conditions = np.zeros((2, len(listmode)), dtype=np.bool)
    #is tag
    conditions[0] = is_event(listmode) == False
    #is time
    conditions[1] = ((listmode >> 28) & 0Xe) == 8
    return np.all(conditions, axis=0)

def get_timetag_indices(listmode):
    """ timetag_indices: 1d indices into petlink data list
    return: np.nonzero(is_timetag(listmode))[0] """
    return np.nonzero(is_timetag(listmode))[0]

def get_timeslice_indices(timetags, timestep):
    """ timeslice_indices: where to split the data for the timeslices.
    each timeslice will have at least timestep timetags (last slice possibly more) 
    the petlink timetags are for preceding data, at time 1 ms, ..., time N ms (end of scan)
    return timetags[timestep-1: -timestep:timestep] """
    return timetags[timestep: -timestep:timestep]

def is_prompt(events):
    """ a bool array indicating prompts in LM events"""
    return (events >> 30) & 1 == 1

def get_prompts(events):
    """ takes numpy array of 32 bit listmode data with events and only returns the prompts (that is, delays removed)"""
    return events[is_prompt(events)]

def is_delay(events):
    """ a bool array indicating delays in LM events"""
    return is_prompt(events) == False

def get_delays(events):
    """ takes numpy array of 32 bit listmode data with events and only returns the delays (that is, prompts removed)"""
    return events[is_delays(events)]

def get_bin_address(events):
    """ extract the 1D bin adress bits of LM events ,
    0X3FFFFFFF is in binary: 00111...111 (32 bits, 2 0s in front)
    return events & 0X3FFFFFFF """
    return events & 0X3FFFFFFF

def binary_to_string(number):
    return '{0:b}'.format(number)

#Thank you Siemens 4 ring mCT walkthrough. (code demonstrating unravel details)
def get_tof_lor_bins(events, shape):
    """ get_tof_lor_bins: get bin addresses and unravel to shape
    extract [tofbin,mi,tx,robin] bin
    formula: event = 1*robin+c1*tx+c1*c2*mi+c1*c2*c3*tof 
    robin = ro - minimum(ro)
    input: events (tags and prompt bits 31,30 can remain in input, they are removed) (np.uint32)
        shape (of projection space), 
    return: coordinates """
    
    ba = get_bin_address(events)
    coordinates = np.unravel_index(ba, shape, order='C')
    coordinates = np.squeeze(np.array(coordinates))
    return coordinates

#Thank you Siemens 4 ring mCT walkthrough. (code demonstrating unravel details)
def get_histogram_bins(events, shape,verbose=False):
    """ projection_bins: get bin addresses and unravel to shape
    extract [tofbin,mi,tx,robin] bin
    formula: event = 1*robin+c1*tx+c1*c2*mi+c1*c2*c3*tof 
    robin = ro - minimum(ro)
    input: events (tags and prompt bits 31,30 can remain in input, they are removed) (np.uint32)
        shape (of projection space), 
    return: coordinates """

    ba = np.array(get_bin_address(events), dtype = int)
    sizes = np.cumprod(np.flip(shape))
    SINOW = sizes[0]
    sino_sz = sizes[1]
    mich_sz = sizes[2]
    
    tof = ba // mich_sz    
    sn = (ba - (tof * mich_sz)) // sino_sz    
    si = (ba - (tof * mich_sz)) - (sn * sino_sz)    
    tx_ang = si // SINOW
    robin = si - (tx_ang * SINOW)

    return np.array([tof,sn,tx_ang,robin])