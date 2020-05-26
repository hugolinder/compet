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
    return transform_toflor(toflor_bins_a, natural_units, axes, image_bins_a, image_bins_b, scan_dict, verbose=verbose)

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
    (TOF LOR bins is a sequence of sequences [tofbins,michelogram bins, transaxial angle, radial offset] )  
    return np.array(event_img_bin, dtype=dtype, verbose=False)"""
    
    tofbin, mi,tx, robin = tof_lor_bins

    #reintroduce negative ro

    ringmean_map, segment_map = kex_headers.get_mi_maps()
    ringmean = ringmean_map[mi]
    seg = kex_headers.SEGMENT_OFFSET_MAP[segment_map[mi]]
    tof = kex_headers.TOF_OFFSET_MAP[tofbin]  

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
        ro = arc_radial_offset(kex_headers.NUMBER_OF_PROJECTIONS, kex_headers.NUMBER_OF_CRYSTALS_PER_RING*2)[robin]
        ro_mm = ro*radius_mm
    else:
        ro = kex_headers.get_radial_offset(robin)
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

    #          small timebin 0, center_of_gantry. 4 per tofbin
    # | -5, -4, -3, -2 | -1, 0, +1, +2 | +3, +4, +5, +6 | 
    #________________________________________________________
    # |      -1        |       0       |        +1      |
    #          measured tofbin 0, slightly off center by 1/8 of the tofbin width              
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
        print("robin ", np.unique(robin))
        print("ro", np.unique(ro))
        print("t_point_mm", np.unique(t_point_mm))

    return np.array(event_img_bin)

def arc_radial_offset(ro_size, ro_rad, radius=1):
    ro = np.arange(ro_size) - ro_size//2
    return radius * np.sin(ro*ro_rad)

def arc_robin(ro, ro_size, ro_rad, radius=1):
    sinus = ro/radius
    angle = np.arcsin(sinus) #sin maps negative sinus to negative angles [-pi/2, pi/2]
    robin_float = angle/ro_rad + ro_size//2
    return get_bin(robin_float)

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

def correct_toflor(toflorA, imgA, imgB, ):
    """ correct_toflor: correct toflor for movement of phantom from A to B, what would be the toflor data at B 
    input: assumed to be listmode, and toflor A assumed to be (tof, rm, rd, tx, ro)
    directions of toflor A are preserved in toflor B. 
    This requires handling of the case when data moves outside of the sinogram fov of detectable lor / tofbins
    toflor b is returned in the format (tof, rm, seg, tx, ro) 
    return toflorB  """

    return

def untangle_mi(lorbins, seg_tab, tof=True):
    """ input: lor_bins, segment table, tof= True (lorbins assumed to be [(tofbin), mi, tx, robin]
    return [(tofbin), rm, segbin, tx, robin]"""

    mi,tx, ro = lorbins[-3:]
    mi_rm, mi_seg = get_mi_maps(seg_tab)
    rm = mi_rm[mi]
    seg = mi_seg[mi]

    new_bins = np.array([rm, seg, tx, ro])
    if tof:
        #keep the shape for concatenation with None
        tofbins = lorbins[0,None]
        new_bins = np.concatenate((tofbins, new_bins), axis=0)

    return new_bins

def get_natural_toflor_units(lorbins, scan_dict, verbose=False):
    """ input: lor_bins, segment table, sino_shape    (lorbins assumed to be [(tof), mi, txbin, robin]
    time of flight is assumed based on the sinogram shape. is_tof = len(lorbins) == 4

    return dictionary with rm, rd, ro and tx (and tof) """

    is_tof = len(lorbins) == 4

    sino_shape = scan_dict['toflor shape'] if is_tof else scan_dict['lor shape'] 

    seg_tab = scan_dict['segment table']
    temp = untangle_mi(lorbins, seg_tab, tof=is_tof)

    rm, segbin, txbin, robin = temp[-4:]
    if is_tof:
        tofbin = temp[0]
        omap = get_offset_map(scan_dict['number of tof bins'])
        tofo = omap[tofbin]
        if verbose:
            print("tofo", tofo)
            print("tofbin_s", scan_dict['tof time'])
            print("tof_offset_s", scan_dict['tof offset'])
            print("lightspeed", scan_dict['flightspeed'])
        tof_len = (tofo*scan_dict['tof time'] + scan_dict['tof offset'])*scan_dict['flightspeed']

    omap = get_offset_map(scan_dict['number of segments'])
    sego = omap[segbin]
    if scan_dict['arc']:
        ro_len = arc_radial_offset(sino_shape[-1], scan_dict['ro rad'], radius=scan_dict['radius'])[robin] 
    else:
        ro_len = (robin + scan_dict['ro offset'])*scan_dict['ro mm']

    tx_rad = (txbin + scan_dict['tx offset'])* np.pi / sino_shape[-2]
    rd_len = sego*scan_dict['span']*scan_dict['rd mm']
    rm_len = rm*scan_dict['rm mm']

    lordict = {"rm": rm_len, 
               "rd": rd_len, 
               "tx": tx_rad,
               "ro": ro_len}
    if is_tof:
        lordict["tof"] = tof_len
    return lordict

def get_bins(float_data):
    return np.round(float_data).astype(int)

def get_toflor_bins(natural_toflor, scan_dict, verbose=False):
    """ natural toflor in mm and radians. requires vector input in natural toflor"""
    tofbin_time = natural_toflor["tof"]/(constants.speed_of_light*1000)-scan_dict["tof offset"]
    tofbin_offset = tofbin_time/scan_dict['tof time']
    tofbin_offset = get_bins(tofbin_offset)
    delays = np.abs(tofbin_offset) > scan_dict["max tof offset"]
    tof_bin = np.empty_like(tofbin_offset)
    tof_bin[delays] = scan_dict["number of tof bins"]
    tof_bin[delays == False] = get_offset_map_inverse(scan_dict['number of tof bins'])[tofbin_offset[delays == False]]
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
    tx_bin = get_bins((natural_toflor["tx"]/scan_dict["tx rad"]) -scan_dict["tx offset"])
    ro = natural_toflor['ro']
    is_ro_fov = (ro < scan_dict['radius'])*(-ro < scan_dict['radius'])
    fov_ro = ro[is_ro_fov]
    angles = np.arcsin(fov_ro/scan_dict["radius"])
    flip = angles > np.pi/2
    angles[flip] = angles[flip] - np.pi
    ro_bin = np.zeros_like(ro)-1
    ro_bin[is_ro_fov] = angles/scan_dict['ro rad'] 
    ro_bin = get_bins(ro_bin-scan_dict['ro offset'])
    is_ro_bin = (ro_bin > -1)*(ro_bin <scan_dict['ro size'])
    conditions = [is_ro_fov, is_ro_bin, delays==False,is_seg_fov, is_mi_fov]
    is_fov = np.all(conditions, axis=0)
    if verbose:
        print("conditions:")
        labels = ["radius", "ro bin", "TOF", "seg", "mi", "all"]
        conditions.append(is_fov)
        N = len(is_fov)
        for lab, con in zip(labels, conditions):
            print(lab + " passed {} out of {}".format(np.sum(con), N))
    return np.array([tof_bin, mi_bin, tx_bin, ro_bin]), is_fov

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