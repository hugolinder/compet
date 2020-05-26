import numpy as np

class MC(type):
    def __str__(self):
        name = "PETModel\n"
        name += "NUMBER_OF_RINGS:=" + str(self.NUMBER_OF_RINGS) + "\n"
        name +="NUMBER_OF_CRYSTALS_PER_RING:=" + str(self.NUMBER_OF_CRYSTALS_PER_RING) + "\n"
        name += "CRYSTAL_RADIUS_mm:=" + str(self.CRYSTAL_RADIUS_m) + "\n"
        name += "NUMBER_OF_TOFBINS:=" + str(self.NUMBER_OF_TOFBINS) + "\n"
        name += "TOF BIN TIME [s] :=" + str(self.TOF_BIN_TIME_s) + "\n"
        name += "TOF OFFSET [s]:=" + str(self.TOF_OFFSET_s)

        return name

class PETModel(object):
    """description of class
    describes the constants of a cylindrical scanner"""
    __metaclass__ = MC
    def __init__(self):
        self.NUMBER_OF_RINGS = 55
        self.NUMBER_OF_CRYSTALS_PER_RING=672
        self.NUMBER_OF_TOFBINS = 13

                #header file info, relating bins and mm
        #norm /sino / LM file hdr
        # z y x
        # header file
        # "transaxial FOV diameter (cm) :=70.4"
        # mCT brochure / Jakoby et al
        # "the industry’s only 78 cm bore ""

        # e7tools vg60 gm_check 1104 (model 1104)
        # ...
        # crystalRadius()=42.76cm
        # ... 
        # sinogramDepthOfInteraction()=0.67cm
        # LORDepthOfInteraction()=0.96cm
        # ...
        #  tofOffset()=0.039ns
        # tofBinSize()=0.3125ns
        # ... 
        self.CRYSTAL_RADIUS_mm = 42.76*10
        self.LOR_DEPTH_OF_INTERACTION_mm= 9.6
        self.SINOGRAM_DEPTH_OF_INTERACTION_mm=6.7
        self.TOF_BIN_TIME_s = 312.5*1e-12
        self.TOF_OFFSET_s = self.TOF_BIN_TIME_s / 8.0

    #methods
    def number_of_views(self, transaxial_compression=2):
        """ set_number_of_views: set transaxial angles"""
        nangles = self.NUMBER_OF_CRYSTALS_PER_RING //2
        return nangles // transaxial_compression

    def offset_map(self,size):
        """ offset map: return [-0,+1,-1, +2, -2,...] of length size. """
        offsets = np.arange(size)
        offsets[0: :2] *= -1
        offsets = np.cumsum(offsets)
        return offsets


        
        
        