import numpy as np 
from PETModel import PETModel

class ScanParams(object):
    """description of class
    describes the data constants for a PET scan"""
    def __init__(self):
        """ default scanner parameters from kex data headers """
        self.model = PETModel()
        self.TRANSAXIAL_COMPRESSION = 2
        self.set_number_of_views()

        #radial elements
        self.NUMBER_OF_PROJECTIONS = 400

        #span
        self.AXIAL_COMPRESSION = 11
        self.MAXIMUM_RING_DIFFERENCE = 49
        self.set_number_of_segments()
        self.set_segment_table()
        #count sinograms per segment
        self.NUMBER_OF_SINOGRAMS = np.sum(self.SEGMENT_TABLE)

        self.HISTOGRAM_DIMENSION_TITLES= np.array([ "TOF bin", 
                                                    "michelogram bin", 
                                                    "transaxial angle bin", 
                                                    "radial offset bin"])
        self.HISTOGRAM_SHAPE_NO_DELAYS =    (self.model.NUMBER_OF_TOFBINS,
                                            self.NUMBER_OF_SINOGRAMS,
                                            self.NUMBER_OF_VIEWS,
                                            self.NUMBER_OF_PROJECTIONS)
        #include delays bin
        self.HISTOGRAM_SHAPE = tuple(np.array(self.HISTOGRAM_SHAPE_NO_DELAYS) + np.array([1,0,0,0]))

        self.LOR_DIMENSION_TITLES= self.HISTOGRAM_DIMENSION_TITLES[1:]
        self.LOR_HISTOGRAM_SHAPE = tuple(np.array(self.HISTOGRAM_SHAPE)[1:])
        
        self.IMAGE_DIMENSION_TITLES = ["z pixel\n(gantry -> bed)", 
                                       "y pixel\n(up -> down)", 
                                       "x pixel\n(left -> right)"]
        width = self.NUMBER_OF_PROJECTIONS //2
        self.IMAGE_SHAPE = (self.SEGMENT_TABLE[0],width, width)

        #datatypes, when siemens saves the files
        self.LISTMODE_DTYPE = np.uint32
        self.SINOGRAM_DTYPE = np.ushort
        self.NORM_DTYPE = np.float32


    

    def segment_minimum_maps(self):
        """ segment_minimum_maps: the minimum michelogram index and ringmean in each index
        return minimum_mi, minimum_ringmean"""
        
        minimum_ringmean = self.minimum_ringmean(np.arange(self.NUMBER_OF_SEGMENTS))
        minimum_mi = np.cumsum(self.SEGMENT_TABLE) - self.SEGMENT_TABLE
        return minimum_mi, minimum_ringmean

    def mi_maps(self):
        """mi_maps: from 1 dimensional michelogram index mi to 2 underlying dimensions
        return mi_ringmean_map, mi_segment_map"""
        
        mi_segment = np.empty(self.NUMBER_OF_SINOGRAMS,dtype=int) 
        mi_ringmean = np.empty_like(mi_segment)
        _, minimum_ringmean = self.segment_minimum_maps
        mi_b= 0
        for segment_number, sinogram_count in enumerate(self.SEGMENT_TABLE):
            mi_a = mi_b
            mi_b += sinogram_count
            mi_segment[mi_a:mi_b] = segment_number
            mi_ringmean[mi_a:mi_b] = np.arange(sinogram_count) + minimum_ringmean[segment_number] 
            
        return mi_ringmean, mi_segment
   
    def segment_ringmean(self, mi):
        """ segment_ringmean: converts 1D michelogram index to segment number and ring mean.
        precalculated maps are probably faster 
        return segment_number, ringmean """
        upper_bounds = np.cumsum(self.SEGMENT_TABLE)
        segment_number = np.searchsorted(upper_bounds, mi+1, side='left')
        if segment_number == 0:
            ringmean = mi
        else:
            index_in_segment = mi - upper_bounds[segment-1]
            ringmean = index_in_segment + (self.NUMBER_OF_DIRECT_SINOGRAMS - self.SEGMENT_TABLE)//2
        return segment_number, ringmean


    def mi_bounds(self,segment_number):
        """ mi_bounds: mi_low:mi_high gives all mi indices within the segment
        input_segment_number (0,1,2,3,...)
        return mi_low, mi_high"""
        mi_low = np.cumsum(self.SEGMENT_TABLE[:segment_number-1])[-1]
        mi_high = mi_low + self.SEGMENT_TABLE[segment_number]
        return mi_low, mi_high   

    def to_string(self):
        str = self.model.to_string()

