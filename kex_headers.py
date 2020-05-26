""" data parameters from KEX header files """ 

import numpy as np
from scipy import constants

#model
NUMBER_OF_RINGS = 55
NUMBER_OF_CRYSTALS_PER_RING=672
NUMBER_OF_TOFBINS = 13

CRYSTAL_RADIUS_MM = 42.76*10
LOR_DEPTH_OF_INTERACTION_MM= 9.6
SINOGRAM_DEPTH_OF_INTERACTION_MM=6.7
TOF_BIN_TIME_S = 312.5*1e-12
TOF_OFFSET_S = TOF_BIN_TIME_S / 8.0

#sinogram header
MM_PER_PIXEL = np.array([2.027, 4.07283,4.07283])
MM_PER_RO = 2.005


def get_offset_map(size):
    """ offset map: return [-0,+1,-1, +2, -2,...] of length size. """
    offsets = np.arange(size)
    offsets[0: :2] *= -1
    offsets = np.cumsum(offsets)
    return offsets

TOF_OFFSET_MAP = get_offset_map(NUMBER_OF_TOFBINS)

#scan params
TRANSAXIAL_COMPRESSION = 2
NUMBER_OF_VIEWS = 168
NUMBER_OF_PROJECTIONS = 400

def get_radial_offset(robin):
    return robin - NUMBER_OF_PROJECTIONS //2

AXIAL_COMPRESSION = 11
MAXIMUM_RING_DIFFERENCE = 49
NUMBER_OF_SEGMENTS = 9
SEGMENT_TABLE = np.array([  109,
                    97, 97,
                    75, 75,
                    53, 53,
                    31, 31])

SEGMENT_OFFSET_MAP = get_offset_map(NUMBER_OF_SEGMENTS)
NUMBER_OF_SINOGRAMS = np.sum(SEGMENT_TABLE)




#data storage
HISTOGRAM_DIMENSION_TITLES= np.array([ "TOF", 
                                            "michelogram", 
                                            "transaxial angle", 
                                            "radial offset"])
HISTOGRAM_UNIT = "bin"
HISTOGRAM_SHAPE_NO_DELAYS =    (NUMBER_OF_TOFBINS,
                                    NUMBER_OF_SINOGRAMS,
                                    NUMBER_OF_VIEWS,
                                    NUMBER_OF_PROJECTIONS)
#include delays bin
HISTOGRAM_SHAPE = tuple(np.array(HISTOGRAM_SHAPE_NO_DELAYS) + np.array([1,0,0,0]))

LOR_DIMENSION_TITLES= HISTOGRAM_DIMENSION_TITLES[1:]
LOR_HISTOGRAM_SHAPE = tuple(np.array(HISTOGRAM_SHAPE)[1:])
        
IMAGE_DIMENSION_TITLES = ["z", "y", "x"]
IMAGE_DIMENSION_DIRECTIONS = ["from gantry to bed", 
                              "from up to down",
                              "from left to right"]
IMAGE_UNIT = "pixel"
width = NUMBER_OF_PROJECTIONS //2
IMAGE_SHAPE = (SEGMENT_TABLE[0],width, width)

LISTMODE_DTYPE = np.uint32
SINOGRAM_DTYPE = np.ushort
NORM_DTYPE = np.float32

# easier to pass between functions
SCAN_DICT={'segment table': SEGMENT_TABLE,
                'number of segments': NUMBER_OF_SEGMENTS,
                'number of views': NUMBER_OF_VIEWS,
                'max tof offset': max(TOF_OFFSET_MAP),
                'number of tof bins': NUMBER_OF_TOFBINS,
                'number of sinograms': NUMBER_OF_SINOGRAMS,
                'mi size': NUMBER_OF_SINOGRAMS,
                'span': AXIAL_COMPRESSION,
                'toflor shape': HISTOGRAM_SHAPE,
                'lor shape': LOR_HISTOGRAM_SHAPE,
                'rm mm': MM_PER_PIXEL[0],
                'rd mm': MM_PER_PIXEL[0]*2,
                'max rd': MAXIMUM_RING_DIFFERENCE,
                'max seg': MAXIMUM_RING_DIFFERENCE//AXIAL_COMPRESSION,
                'tof time': TOF_BIN_TIME_S,
                'tof offset': TOF_OFFSET_S,
                'flightspeed': constants.speed_of_light*1000,
                'tx offset': TRANSAXIAL_COMPRESSION/2,
                'tx rad': np.pi / NUMBER_OF_VIEWS,
                'tx size': NUMBER_OF_VIEWS,
                'ro offset': -NUMBER_OF_PROJECTIONS //2,
                'ro rad': np.pi/NUMBER_OF_CRYSTALS_PER_RING, 
                'ro mm': MM_PER_RO,
                'ro size': NUMBER_OF_PROJECTIONS,
                'radius': CRYSTAL_RADIUS_MM + LOR_DEPTH_OF_INTERACTION_MM + SINOGRAM_DEPTH_OF_INTERACTION_MM,
                'arc':True, 
                'origin pixel': [0, IMAGE_SHAPE[1]//2, IMAGE_SHAPE[2]//2]}


def get_mi_maps():
        """mi_maps: from 1 dimensional michelogram index mi to 2 underlying dimensions
        return mi_ringmean_map, mi_segment_map"""
        
        mi_segment = np.empty(NUMBER_OF_SINOGRAMS,dtype=int) 
        mi_ringmean = np.empty_like(mi_segment)

        minimum_ringmean = (SEGMENT_TABLE[0] - SEGMENT_TABLE ) //2
        mi_b= 0
        for segment_number, sinogram_count in enumerate(SEGMENT_TABLE):
            mi_a = mi_b
            mi_b += sinogram_count
            mi_segment[mi_a:mi_b] = segment_number
            mi_ringmean[mi_a:mi_b] = np.arange(sinogram_count) + minimum_ringmean[segment_number] 
            
        return mi_ringmean, mi_segment

def arc_correction_map():


    return

    

