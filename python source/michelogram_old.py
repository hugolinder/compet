import numpy as np

SEGMENT_TABLE = np.array([  109,
                            97, 97,
                            75, 75,
                            53, 53,
                            31, 31])

def get_number_of_segments(number_of_rings=55, maximum_ring_difference=49,span=11):
    maximum_direct_ring_difference = (span-1)//2
    number_of_positive_segments = (maximum_ring_difference - maximum_direct_ring_difference ) // span
    return 1+2*number_of_positive_segments

def get_segment_table(number_of_rings=55, maximum_ring_difference=49, span=11):
    """get_segment_table: sinogram count per segment (0,+1,-1,+2,-2, ... )
    same for positive and negative (oblique) segments (at same magnitude)
    direct segment is special
    return segment_table"""
    number_of_segments = get_number_of_segments(number_of_rings,maximum_ring_difference)
    segment_table = np.empty(number_of_segments, dtype=int)
    sinogram_count = number_of_rings*2 - 1
    #direct segment
    segment_table[0] = sinogram_count
    sinogram_count -= span+1
    #remaining oblique segments
    obliques = number_of_segments -1
    indices = np.arange(obliques)
    half_indices = indices // 2
    segment_table[1+indices] = np.arange(sinogram_count,0,-span*2)[half_indices]
    return segment_table

def get_minimum_ringmean(segment,segment_table=None):
    """ get_minimum_ringmean: in segment"""
    if segment_table is None:
            segment_table = SEGMENT_TABLE
    return (segment_table[0] - segment_table[segment])//2

def get_segment_minimum_maps(segment_table=None):
    """ get_segment_minimum_maps: the minimum michelogram index and ringmean in each index
    return minimum_mi, minimum_ringmean"""
    if segment_table is None:
            segment_table = SEGMENT_TABLE
    minimum_ringmean = get_minimum_ringmean(segment_table, np.arange(len(segment_table)))
    minimum_mi = np.cumsum(segment_table) - segment_table
    return minimum_mi, minimum_ringmean

def get_number_of_sinograms(segment_table):
    return np.cumsum(segment_table)

def get_mi_maps(segment_table=None):
        """get_mi_maps: from 1 dimensional michelogram index mi to 2 underlying dimensions
        return mi_ringmean_map, mi_segment_map"""
        if segment_table is None:
            segment_table = SEGMENT_TABLE

        mi_segment = np.empty(get_number_of_sinograms(segment_table),dtype=int) 
        mi_ringmean = np.empty_like(mi_segment)
        _,minimum_ringmean = get_segment_minimum_maps(segment_table)
        mi_b= 0
        for segment_number, sinogram_count in enumerate(segment_table):
            mi_a = mi_b
            mi_b += sinogram_count
            mi_segment[mi_a:mi_b] = segment_number
            mi_ringmean[mi_a:mi_b] = np.arange(sinogram_count) + minimum_ringmean[segment_number]
        return 

def get_mi_bounds(segment_number, segment_table=None):
    """ mi_bounds: mi_low:mi_high gives all mi indices within the segment
    input_segment_number (0,1,2,3,...)
    return mi_low, mi_high"""
    if segment_table is None:
        segment_table = SEGMENT_TABLE

    mi_low = np.cumsum(segment_table[:segment_number-1])[-1]
    mi_high = mi_low + segment_table[segment_number]
    return mi_low, mi_high

def get_segment_ringmean(mi,segment_table=None):
    """ segment_ringmean: converts 1D michelogram index to segment number and ring mean.
    precalculated maps are probably faster 
    return segment_number, ringmean """
    if segment_table is None:
        segment_table = SEGMENT_TABLE
    upper_bounds = np.cumsum(segment_table)
    segment_number = np.searchsorted(upper_bounds, mi+1, side='left')
    if segment_number == 0:
        ringmean = mi
    else:
        index_in_segment = mi - upper_bounds[segment-1]
        ringmean = index_in_segment + (segment_table[0] - SEGMENT_TABLE)//2
    return segment_number, ringmean


