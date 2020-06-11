import numpy as np


def get_offset_map(size):
    """ offset map: return [-0,+1,-1, +2, -2,...] of length size. size should be odd
    return omap"""
    omap = np.arange(size)
    omap[0: :2] *= -1
    #[-0,1,-2,+3,...]
    omap = np.cumsum(omap)
    #[ 0,1,-1,+3...]
    return omap

def get_offset_map_inv(size):
    """ inverse of offset map: negative indices get mapped to where they belong
    let x = (size-1)//2 (size is odd)
    map from [0,1,....,+x,-x,...,-1] to [0,1,3,5,...,2x-1,2x... ,2] 
    the idea is that imap[omap] = np.arange(size), 
    where omap = get_offset_map(size)
    return imap"""
    imap = np.zeros(size, dtype=int)
    x = (size-1)//2
    rng = np.arange(x,dtype=int)*2
    imap[1:(x+1)] = rng +1
    imap[-1:-(x+1):-1] = rng + 2
    return imap

def get_segment_table(num_rings=1, max_rd=1, span=1, compression=None):
    """ get_segment_table: indicates number of sinogram bins per segment
    segments are ordered 0,+1,-1,+2,-2, ... 
    thus segment_table[0] has the most sinograms
    input: ring_num, max_rd, span, compression object
    return sino_per_segment """
    if compression is not None:
        num_rings = compression.scanner.num_rings
        max_rd = compression.get_max_ring_diff()
        span= compression.span_num
    #include negative, positive, and 0 ring differences
    rd_size = 2*max_rd+1
    #this is a reason why span should be odd, it should divide an odd number
    num_segments = rd_size // span 
    sino_per_segment = np.zeros(num_segments, dtype=int)
    direct_size = num_rings
    
    if span > 1: direct_size += num_rings-1  #add halfplanes
        
    segment_size = direct_size
    for s in range(num_segments):
        sino_per_segment[s] = segment_size
        #direct to oblique span split, from - to + span 1D, from + to -
        subtract = (span+1)/2 if s == 0 else (span if s % 2 ==0 else 0)
        if span > 1:  subtract *= 2   #remove halfplanes
        segment_size -=subtract
    return sino_per_segment

def get_mi_maps(seg_tab=None, compression=None, onemap=True, like_odlpet=False):
    """mi_maps: from 1 dimensional michelogram index mi to 2 underlying dimensions 
    with like_odlpet: segment index in [-, ..., 0, ..., +] and axial index [0, ...]
    alternatively, 2*ringmean and segment number [0, ...]
    
    input: seg_tab, compression, onemap=True, like_odlpet=True
    if seg_tab is None:   seg_tab = get_segment_table(compression=compression)
    tup = (mi_seg, mi_rm)
    rturn np.array(tup).T if onemap else tup"""
    if seg_tab is None:     seg_tab = get_segment_table(compression=compression)
    num_sino = np.sum(seg_tab)
    mi_seg = np.empty(num_sino,dtype=int) 
    mi_rm = np.empty_like(mi_seg)
    
    if like_odlpet: omap = get_offset_map(len(seg_tab))
    
    mi_b= 0
    for seg_num, seg_size in enumerate(seg_tab):
        mi_a = mi_b
        mi_b += seg_size
        mi_seg[mi_a:mi_b] = omap[seg_num] if like_odlpet else seg_num
        mi_rm[mi_a:mi_b] = np.arange(seg_size) + (0 if like_odlpet else seg_tab[0] - seg_tab[seg_num] )
    tup = (mi_seg, mi_rm)
    return np.array(tup).T if onemap else tup
