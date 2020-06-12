import numpy as np
from scipy import constants
import kex_headers

def circle_recon(tof_lor_bins, scan_dict=None):
    tof_lor_bins[0] = 0
    return get_img_bins2(tof_lor_bins, timealign=False)

def transform_toflor(toflor_bins_a, natural_toflor_a, axes, image_a, image_b, scan_dict, verbose=False):
    """ image_a, image_b in mm coordinates. axes R,S,T in dictionary """
    ro = np.sum(image_b[1:]*axes['R'], axis=0) # projection on R unit axis
    natural_toflor_b = {'ro': ro} 
    natural_toflor_b['tx'] = natural_toflor_a['tx'] # preserve tx angle
    rd = np.zeros_like(natural_toflor_b['ro']) # new ring difference
    T_tx = np.sqrt(np.sum(axes['T'][1:]**2, axis=0))
    T_z = axes['T'][0]
    safe_div = T_z != 0 # safe for T_tx/T_z
    width=2*np.sqrt(np.maximum(scan_dict['radius']**2 -ro[safe_div]**2,0)) # right triangle
    tilt_constant = T_tx[safe_div]/T_z[safe_div] # =width/rd, when rd != 0
    rd[safe_div] =  width / tilt_constant
    natural_toflor_b['rd'] = rd
    img_s = np.sum(image_b[1:]*axes['S'], axis=0) # projction on S axis
    T_s = np.sum(axes['T'][1:]*axes['S'], axis=0) # projection on S axis
    t = img_s / T_s # t*T_s = img_s
    natural_toflor_b['tof'] = t*2 # length of flight
    midpoint = image_b - t*axes['T']
    natural_toflor_b['rm'] = midpoint[0]
    return get_toflor_bins(natural_toflor_b, scan_dict, verbose) # from natural units to data bins

def move_translation_projection(toflor_bins, xyz_translation=[0,0,0], scan_dict=None, verbose=False):
    """ Linear transformation by keeping LOR directions fixed (axial and transaxial angles) """
    if scan_dict is None: scan_dict = kex_headers.SCAN_DICT
    toflor_a_natural = get_natural_toflor_units(toflor_bins, scan_dict)
    tx = toflor_a_natural['tx']
    toflor_b_natural = {'tx': tx}
    ro_a =  toflor_a_natural['ro']
    kwargs = {'ro': ro_a, 'rd': toflor_a_natural['rd'], 'radius': scan_dict['radius']}
    cos_sin = get_cos_sin(tx,**kwargs)
    proj_matrix = image_to_data_matrix(**cos_sin)
    subscripts = "ijn,j->in" #could also be used for einsum paths
    if proj_matrix.ndim < 3: # a single event
        subscripts = subscripts.replace("n","")
    trz_translation = np.einsum(subscripts, proj_matrix, xyz_translation)
    toflor_b_natural['tof'] = toflor_a_natural['tof'] + 2*trz_translation[0]
    toflor_b_natural['ro'] = ro_a + trz_translation[1]
    toflor_b_natural['rm'] = toflor_a_natural['rm'] + trz_translation[2] 
    kwargs = {'ro': toflor_b_natural['ro'], 'radius': scan_dict['radius']}
    for s in ['cos', 'sin']:
        cos_sin.pop(s+'_tx')
    toflor_b_natural['rd'] = get_rd(**kwargs, **cos_sin)
    if verbose:
        print("einsum subscripts", subscripts)
        diffs = {k: toflor_b_natural[k]-v for k, v in toflor_a_natural.items()}
        diff_sums = {k: np.sum(v) for k, v in diffs.items()}
        for k, v in diff_sums.items():
            print("diff sum {} = {}".format(k,v))
        print("diffs rd = {}".format(diffs['rd']))
        for k, v in zip('ab',[toflor_a_natural, toflor_b_natural]):
            uniques = np.unique(v['rd'].astype(int))
            print("unique rd_{} = {}".format(k,uniques))
    return  get_toflor_bins(toflor_b_natural, scan_dict, verbose=False)

def move_translation(toflor_bins_a, translation, scan_dict=None, verbose=False):
    if scan_dict is None: scan_dict = kex_headers.SCAN_DICT
    natural_units = get_natural_toflor_units(toflor_bins_a,kex_headers.SCAN_DICT)
    axes = get_lor_axes(natural_units, scan_dict['radius'])
    midpoint = np.concatenate(([natural_units['rm']], natural_units['ro']*axes['R']), axis=0)
    image_bins_a = (midpoint + natural_units['tof']*axes["T"]/2)
    image_bins_b = image_bins_a + np.expand_dims(translation, 1) 
    if verbose:
        print("natural units", natural_units)
        print("axes", axes)
        print("midpoint", midpoint)
        print("image bins a", image_bins_a)
        print("image bins b", image_bins_b)
    args = (toflor_bins_a, natural_units, axes, image_bins_a, image_bins_b, scan_dict)
    return transform_toflor(*args, verbose=verbose)

def data_to_image_matrix(cos_tx, sin_tx, cos_ax, sin_ax):
    """ relating tof, ro, rm to x, y, z , A (tof, ro, rm) = (x,y,z)
    A = [T,R,Z] (axes) """
    N = np.size(cos_tx)
    A = np.zeros((3,3,N))
    A = np.squeeze(A)
    A[:,0] = cos_ax*sin_tx, -cos_ax*cos_tx, sin_ax #T
    A[0:2,1] = cos_tx, sin_tx #R
    A[2,2] = 1 #Z
    return A

def image_to_data_matrix(cos_tx, sin_tx, cos_ax, sin_ax):
    N = np.size(cos_tx)
    B = np.zeros((3,3,N))
    B = np.squeeze(B)
    B[0,:2] = sin_tx/cos_ax, -cos_tx/cos_ax
    B[1,:2] = cos_tx, sin_tx
    B[2,:2] = -sin_ax*sin_tx/cos_ax, sin_ax*cos_tx/cos_ax
    B[2,2] = 1
    return B

def axes_matrix(tx, **kwargs):
    return data_to_image_matrix(**get_cos_sin(tx, **kwargs))

def axes_matrix_inverse(tx, **kwargs):
    return image_to_data_matrix(**get_cos_sin(tx, **kwargs))

def get_cos_sin(tx, **kwargs):
    cos_ax, sin_ax = get_cos_sin_ax(**kwargs)
    cos_tx = np.cos(tx)
    sin_tx = np.sin(tx)
    names= "cos_tx, sin_tx, cos_ax, sin_ax".split(", ")
    values = [cos_tx, sin_tx, cos_ax, sin_ax]
    return dict(zip(names, values))

def get_rd(ro, radius, cos_ax, sin_ax):
    return 2*np.sqrt(radius**2 - ro**2)*sin_ax/cos_ax

def get_cos_sin_ax(ro, radius, rd):
    """ return cos_ax, sin_ax """
    width = 2*np.sqrt(radius**2 - ro**2)
    L = np.sqrt(width**2 + rd**2)
    return width/L, rd/L 

def get_img_bins2(tof_lor_bins, timealign=True, verbose=False, arc=True):
    scan_dict = kex_headers.SCAN_DICT.copy()

    if not timealign:
        scan_dict['tof offset'] = 0
    if not arc:
        scan_dict['arc']=False
    natural_units = get_natural_toflor_units(tof_lor_bins,scan_dict, verbose=verbose)

    axes = get_lor_axes(natural_units, scan_dict['radius'])
    #natural_units['ro'] *= 0 #testing
    #natural_units['tof'] *= 0 #testing
    #axes['T'] *= 0 #testing
    midpoint = np.concatenate(([natural_units['rm']], natural_units['ro']*axes['R']), axis=0)
    origin = kex_headers.SCAN_DICT['origin pixel']

    return (midpoint + natural_units['tof']*axes["T"]/2) / np.expand_dims(kex_headers.MM_PER_PIXEL, axis=1) + np.expand_dims(origin, axis=1) 

def get_image_bins(tof_lor_bins, timealign=True, verbose=False, arc=True):
    """ image_bins: vectorized transformation from Time Of Flight Line Of Respone (TOF LOR) data to image bin [pixel z y x] 
    for the 4-ring PET scanner mCT defined by kex_headers.py
    input: tof_lor_bins, offset=True,  verbose=False.
    (TOF LOR bins is a sequence of sequences [tof_bins,michelogram bins, transaxial angle, radial offset] )  
    return np.array(event_img_bin, dtype=dtype, verbose=False)"""
    
    tof_bin, mi,tx, ro_bin = tof_lor_bins

    #reintroduce negative ro

    ringmean_map, segment_map = kex_headers.get_mi_maps()
    ringmean = ringmean_map[mi]
    seg = kex_headers.SEGMENT_OFFSET_MAP[segment_map[mi]]
    tof = kex_headers.TOF_OFFSET_MAP[tof_bin]  

    #calculate in mm space first, then convert to pixels
    # assume Siemens coordinates, then translate by origin to array
    #decide directions R,S,T from angles
    #As R,S are 2D and transaxial, only T changes between pixel space and mm space
    #tof bin requires mm space

    #precalculate these
    #constant extra cost
    #places upper bound on the number of cos, sin evaluations
    # views in halfturn
    angle_radians = np.pi /kex_headers.NUMBER_OF_VIEWS
    tx_options = np.arange(0,kex_headers.NUMBER_OF_VIEWS,1.0)
    #move to the center of each angle bin
    tx_options += kex_headers.TRANSAXIAL_COMPRESSION/2
    Rx_options = np.cos(tx_options*angle_radians)
    Ry_options = np.sin(tx_options*angle_radians)
    Rx = Rx_options[tx]
    Ry = Ry_options[tx]

    if np.any(Ry < 0):
        print("invalid R direction, y < 0")
    
    # now calculate the detector points A,B of the LOR, and the point C between them, starting with C
    midpoint_mm = np.array([ringmean*kex_headers.MM_PER_PIXEL[0],
                             Ry,
                             Rx])
    #ro_mm = ro*kex_headers.MM_PER_RO
    radius_mm = kex_headers.CRYSTAL_RADIUS_MM #+ kex_headers.LOR_DEPTH_OF_INTERACTION_MM 

    if arc:
        ro = arc_radial_offset(kex_headers.NUMBER_OF_PROJECTIONS, kex_headers.NUMBER_OF_CRYSTALS_PER_RING*2)[ro_bin]
        ro_mm = ro*radius_mm
    else:
        ro = kex_headers.get_radial_offset(ro_bin)
        ro_mm = ro*kex_headers.MM_PER_RO        

    midpoint_mm[1:] *= ro_mm

    #now determine distance from midpoint to detector along S axis
    ro_square = np.power(ro_mm, 2)

    radius_square = np.power(radius_mm, 2)

    #by right triangle
    s_square = radius_square - ro_square
    if (np.any(s_square < 0)):
        print("negative s_square", s_square)
        print("cylinder radius_square", radius_square)
        print("radial offset ro_square", ro_square)
    
    s_cylinder_mm = np.sqrt(s_square)
        
    # segments separated by span rings, average rd in segment is span*segment_number
    #   |    /|   seg +1 
    #   |  /  |   span
    #   |/----|   seg 0

    #span indicates how many rings are traversed axially between segments, and there are 2 pixels (halfplanes) per ring
    half_rd_mm = seg*kex_headers.AXIAL_COMPRESSION*kex_headers.MM_PER_PIXEL[0]
    
    #determine axis T
    #from midpoint to detector at positive s
    T_direction_mm = np.zeros_like(midpoint_mm)
    T_direction_mm[0] = half_rd_mm

    #Sx = Ry
    #Sy = -1*Rx
    S_direction = np.array([-1*Rx, Ry])
    T_direction_mm[1:] = s_cylinder_mm*S_direction

    #sufficient to calculate in 2D
    norms = np.linalg.norm([half_rd_mm, 
                            s_cylinder_mm], axis=0)
    if np.any(norms == 0):
        print("invalid normalization 0")
    T_direction_mm = T_direction_mm / norms
    if np.any(T_direction_mm[2] < 0):
        print("invalid T direction, x < 0")
    
    
    t_s = tof*kex_headers.TOF_BIN_TIME_S
    #correction factor according to Siemens documentation
    # "For TOF, the time direction is t"

    #          small timebin 0, center_of_gantry. 4 per tof_bin
    # | -5, -4, -3, -2 | -1, 0, +1, +2 | +3, +4, +5, +6 | 
    #________________________________________________________
    # |      -1        |       0       |        +1      |
    #          measured tof_bin 0, slightly off center by 1/8 of the tof_bin width              
    if timealign:
        t_s += kex_headers.TOF_OFFSET_S

    flight_m_per_s = constants.speed_of_light
    t_mm = t_s*flight_m_per_s*1e3

    # the "late" photon has to travel t_mm longer, to the mirror point of the event on the other side of origin
    # we are interested in the distance to the origin, not to the mirror point, so the factor 1/2 is used
    # len      L   ,   L             
    # LOR  A-------0-------B
    # pt         x , x'
    # len    L-x , L+x
    # time   t_a , t_b
    # c*t_A = L-x
    # c*t_B = L+x
    # c*(t_B-t_A) = 2*x
    # x = 2 / ( c*(t_B - t_A) )

    t_mm /= 2
    t_point_mm = t_mm*T_direction_mm
    
    event_point_mm = t_point_mm + midpoint_mm     

    #origin position in image array
    origin_array = [0, kex_headers.IMAGE_SHAPE[1]//2, kex_headers.IMAGE_SHAPE[2]//2]

    event_img_bin = []
    # construct dimension by dimension to avoid shape issues
    for k in range(3):
        event_img_bin.append(event_point_mm[k] / kex_headers.MM_PER_PIXEL[k] + origin_array[k])

    if verbose:
        def shape(arr):
            return np.array(arr).shape
        print("--- Shapes ---")
        print("event_point_mm", shape(event_point_mm))
        print("origin_array", shape(origin_array))
        print("---uniques---")
        print("tof offsets", np.unique(tof))
        print("tx", np.unique(tx))
        print("mi", np.unique(mi))
        print("ro_bin ", np.unique(ro_bin))
        print("ro", np.unique(ro))
        print("t_point_mm", np.unique(t_point_mm))
    return np.array(event_img_bin)

def arc_radial_offset(ro_size, ro_rad, radius=1):
    """ returns an array that maps"""
    ro = np.arange(ro_size) - ro_size//2
    return radius * np.sin(ro*ro_rad)

def arc_ro_bin(ro, ro_size, ro_rad, radius=1):
    sinus = ro/radius
    angle = np.arcsin(sinus) #sin maps negative sinus to negative angles [-pi/2, pi/2]
    ro_bin_float = angle/ro_rad + ro_size//2
    return get_bin(ro_bin_float)

def get_bin(bin_floats):
    return np.round(bin_floats).astype(int)

def get_lor_axes(natural_units, radius=1):
    """ note: rd, ro in dict and radius should be in the same units, for example mm. tx angle should be radians.
    return dictionary with axes R,S,T """
    tx_rad = natural_units["tx"]
    Rx = np.cos(tx_rad)
    Ry = np.sin(tx_rad)
    R = np.array([Ry, Rx])
    S = np.array([-1*Rx, Ry ])
    #T is a bit more complicated, requires conversion to mm
    radius_square = radius**2
    ro_square = natural_units["ro"]**2
    s_square = radius_square-ro_square
    #from one detector to the other along s
    s_len = 2*np.sqrt(s_square)

    #T from one detector to the other
    rd = natural_units["rd"]
    T = np.concatenate(([rd], s_len*S), axis=0)
    norms = np.linalg.norm([rd, s_len], axis=0)
    T = T / norms
    
    return {"R":R, "T":T, "S": S}

def get_mi_maps(seg_tab):
    """mi_maps: from 1 dimensional michelogram index mi to 2 underlying dimensions
    return mi_rm, mi_seg    (=mi_ringmean_map, mi_segment_map) """
    num_sino = np.sum(seg_tab)
    mi_seg = np.empty(num_sino, dtype=int)
    mi_rm = np.empty_like(mi_seg)

    min_rm = (seg_tab[0] - seg_tab ) //2
    mi_b= 0
    for seg_num, seg_size in enumerate(seg_tab):
        mi_a = mi_b
        mi_b += seg_size
        mi_seg[mi_a:mi_b] = seg_num
        mi_rm[mi_a:mi_b] = np.arange(seg_size) + min_rm[seg_num] 
            
    return mi_rm, mi_seg

def untangle_mi(mi, seg_sizes):
    """ input: mi, segment table
    return rm, seg """
    max_mi = np.cumsum(seg_sizes) - 1 # per segment
    seg = np.searchsorted(max_mi, mi)
    min_mi = max_mi + 1 - seg_sizes
    rm_offset = mi - min_mi[seg]
    min_rm = (seg_sizes[0] - seg_sizes[seg])//2
    rm = rm_offset + min_rm
    return rm, seg

def get_mi(rm, seg, seg_sizes):
    min_rm = (seg_sizes[0] - seg_sizes[seg])//2
    rm_offset = rm - min_rm
    min_mis = np.cumsum(seg_sizes) - seg_sizes
    return min_mis[seg] + rm_offset

def get_natural_toflor_units(toflor_bins, scan_dict, verbose=False):
    """ input: toflor_bins, segment table, scan_dict 
    return dictionary with tof, ringmean, ring difference, radial offset and transaxial angle  """
    mi = toflor_bins[1]
    rm, seg_bin = untangle_mi(mi, scan_dict['segment table'])
    tx_bin, ro_bin = toflor_bins[-2:]
    tof_bin = toflor_bins[0]
    omap = get_offset_map(scan_dict['number of tof bins']+1) #+1 for delays
    tofo = omap[tof_bin]
    if verbose:
        print("tofo", tofo)
        print("tof_bin_s", scan_dict['tof time'])
        print("tof_offset_s", scan_dict['tof center'])
        print("flightspeed", scan_dict['flightspeed'])
    tof_time = (tofo*scan_dict['tof time'] + scan_dict['tof center'])
    tof_len = tof_time*scan_dict['flightspeed']
    omap = get_offset_map(scan_dict['number of segments'])
    sego = omap[seg_bin]
    if scan_dict['arc']:
        args = [scan_dict[s] for s in ['ro size', 'ro rad', 'radius'] ]
        ro_len = arc_radial_offset(*args)[ro_bin] 
    else:
        ro_len = (ro_bin + scan_dict['ro shift'])*scan_dict['ro mm']
    tx_rad = (tx_bin + scan_dict['tx center'])* np.pi / scan_dict['tx size']
    rd_len = sego*scan_dict['span']*scan_dict['rd mm']
    rm_len = rm*scan_dict['rm mm']
    toflor_natural = {"rm": rm_len, "rd": rd_len, "tx": tx_rad,  "ro": ro_len, 'tof': tof_len}
    return toflor_natural

def get_bins(float_data):
    return np.round(float_data).astype(int)

def get_toflor_bins(natural_toflor, scan_dict, verbose=False):
    """ natural toflor in mm and radians. requires vector input in natural toflor"""
    tof_bin_time = natural_toflor["tof"]/(constants.speed_of_light*1000)-scan_dict["tof center"]
    tof_bin_offset = tof_bin_time/scan_dict['tof time']
    tof_bin_offset = get_bins(tof_bin_offset)
    tof_offset_map = get_offset_map(scan_dict['number of tof bins']+1) #+1 for delays
    delay_offset = max(tof_offset_map)
    delays = np.abs(tof_bin_offset) >= delay_offset
    tof_bin = np.empty_like(tof_bin_offset)
    tof_bin[delays] = scan_dict["number of tof bins"]
    tof_bin[delays == False] = get_offset_map_inverse(scan_dict['number of tof bins'])[tof_bin_offset[delays == False]]
    rd = natural_toflor["rd"]/scan_dict["rd mm"] # from mm units to rings
    segment_offset = rd/(scan_dict["span"] )
    seg_bound = scan_dict['max seg']+1
    is_seg_fov = (segment_offset < seg_bound)*(-segment_offset < seg_bound)
    segment_offset = get_bins(segment_offset)
    segment_bin = get_offset_map_inverse(scan_dict["number of segments"])[segment_offset]
    seg_table = scan_dict["segment table"]
    minimum_ringmean = (seg_table[0] - seg_table[segment_bin])//2
    axial_position = get_bins(natural_toflor["rm"] / scan_dict["rm mm"]) - minimum_ringmean
    upper_bound_mi = np.cumsum(seg_table)  #per segment
    min_mi = upper_bound_mi - seg_table
    mi_bin = min_mi[segment_bin] + axial_position
    is_mi_fov = (mi_bin > (min_mi[segment_bin]-1))*(mi_bin < upper_bound_mi[segment_bin]) 
    tx_bin = get_bins((natural_toflor["tx"]/scan_dict["tx rad"]) -scan_dict["tx center"])
    ro = natural_toflor['ro']
    is_ro_fov = (ro < scan_dict['radius'])*(-ro < scan_dict['radius'])
    fov_ro = ro[is_ro_fov]
    angles = np.arcsin(fov_ro/scan_dict["radius"])
    flip = angles > np.pi/2
    angles[flip] = angles[flip] - np.pi
    ro_bin = np.zeros_like(ro)-1
    ro_bin[is_ro_fov] = angles/scan_dict['ro rad'] 
    ro_bin = get_bins(ro_bin-scan_dict['ro shift'])
    is_ro_bin = (ro_bin > -1)*(ro_bin <scan_dict['ro size'])
    conditions = [is_ro_fov, is_ro_bin, is_seg_fov, is_mi_fov]
    is_fov = np.all(conditions, axis=0)
    if verbose:
        print("conditions:")
        labels = ["radius", "ro bin", "seg", "mi", "all"]
        conditions.append(is_fov)
        N = len(is_fov)
        for lab, con in zip(labels, conditions):
            print(lab + " passed {} out of {}".format(np.sum(con), N))
    return np.array([tof_bin, mi_bin, tx_bin, ro_bin]), is_fov

def get_offset_map(size):
    """ offset map: return [-0,+1,-1, +2, -2,...] of length size. """
    offsets = np.arange(size)
    offsets[0: :2] *= -1
    offsets = np.cumsum(offsets)
    return offsets

def get_offset_map_inverse(size):
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

def get_offset(num):
    mod2 = (num % 2)
    offset = (num + mod2) // 2 #0,1,1,2,2, ...
    sign = 2*mod2-1 #-,+,-,+,...
    return sign*offset

def get_offset_bin(offset):
    mod2 = 1*(offset > 0)
    offset = np.abs(offset)
    return offset*2 - mod2

