import os
from subprocess import Popen, PIPE
import numpy as np
#import matplotlib.pyplot as plt

#Hugo code
#import PythonKEX
from PETModel import PETModel
import pet_images
import kex_headers

NORM_PATH = r"C:/Users/petct/Desktop/KanylPET-Converted/KanylPET-norm.n.hdr"

def paths(is_cannula=True, both=False, tof=True, recon_str="OP"):
    """ paths: to KEX data (Cannula or Cylinder )
    input: is_cannula=True, tof=True, recon_str="OP"
    return listmode_paths, sino_paths, recon_paths """
    #Listmode files

    cylinder = [r"C:/Users/petct/Desktop/CylinderPET-Converted/CylinderPET-LM-00/CylinderPET-LM-00.l",
                     r"C:/Users/petct/Desktop/CylinderPET-Converted/CylinderPET-LM-01/CylinderPET-LM-01.l"]

    cannula = [r"C:/Users/petct/Desktop/KanylPET-Converted/KanylPET-LM-00/KanylPET-LM-00.l",
                     r"C:/Users/petct/Desktop/KanylPET-Converted/KanylPET-LM-01/KanylPET-LM-01.l"]
    if both:
        listmode_paths = cylinder + cannula
    elif is_cannula:
        listmode_paths = cannula
    else:
        listmode_paths = cylinder
        
    #related files
    if tof:
        recon_str = recon_str + "TOF"
    
    sino_paths = []
    recon_paths = []
    for p in listmode_paths:
        #parts = p.split('-LM')
        #sino_paths.append(parts[0] + parts[1] + parts[2][:3] + "-sino.s")
        
        pre_str = p.split('.')[0]
        sino_paths.append(pre_str + "-sino-0.s")
        recon_paths.append(pre_str + "-" + recon_str + "_000_000.v-DICOM")
    
    return listmode_paths, sino_paths, recon_paths

def listmode():
    """ listmode: 
    return kex listmode_data, listmode_paths"""
    listmode_paths = []
    listmode_data = []
    
    for b in [False, True]:
        listmode_paths += paths(is_cannula=b)[0]
        
    for p in listmode_paths:
        with open(p, 'rb') as file:
            listmode_data.append(np.fromfile(file,dtype=kex_headers.LISTMODE_DTYPE))
        
    return listmode_data, listmode_paths

def sinograms():
    """ sinograms: return the KEX sinograms (4D) and the related sino. paths
    the shape order is [tof,michelograms, transaxial angle, radial offset
    return sinograms_4D, sino_paths"""
    sino_paths = []
    sinograms_4D = []
    
    for b in [False, True]:
        sino_paths += paths(is_cannula=b)[1]
    
    for p in sino_paths:
        with open(p, 'rb') as file:
            sinogram_1D = np.fromfile(file,dtype=kex_headers.SINOGRAM_DTYPE)
        sinograms_4D.append(np.reshape(sinogram_1D, newshape=kex_headers.HISTOGRAM_SHAPE, order='C'))
        
    return sinograms_4D, sino_paths
    
def norm_sinograms():
    """ norm_sinograms: returns list of KEX normalized 4D sinograms and the related sino. paths"""
    """ the shape order is [tof,michelograms, transaxial angle, radial offset
    return sinogram_4D, sino_path"""
    norm_sino_folder = r"C:/Users/petct/Desktop/Hugo/Code/PythonMEX/normalised_sinograms"
    subdirs = []
    sino_path = []
    sinogram_4D=[]

    with os.scandir(norm_sino_folder) as it:
        for entry in it:
            if entry.name.endswith(".s"):
                s_path = norm_sino_folder + r"/" + entry.name
                sino_path.append(s_path)
                #print("entry name", entry.name)

                with open(s_path, 'rb') as file:
                    sinogram_1D = np.fromfile(file,dtype=kex_headers.NORM_DTYPE)
                sinogram_4D.append(np.reshape(sinogram_1D, newshape=kex_headers.HISTOGRAM_SHAPE_NO_DELAYS, order='C'))
    
    return sinogram_4D, sino_path

def get_norm_factors():
    return intermediates(only_norm=True)

def intermediates(only_norm=False):
    """ intermediates: returns a list with 3D sinograms [emis, emis_corr, norm3d] for each PET gate in KEX data. Also returns the paths
    return sinograms_3D, paths """
    intermediate_folder = r"C:/Users/petct/Desktop/Hugo/Code/PythonMEX/normalised_sinograms/intermediates"
    subdirs = []
    with os.scandir(intermediate_folder) as it:
        for entry in it:
            if entry.is_dir():
                #print("dir entry name:",  entry.name)
                subdirs.append(entry.name)     
    #common for all, relevant data from header files
    #number format := float
    #number of bytes per pixel := 4
    #4*8=32
    #... but the file sizes vary, maybe they pad the "z elements" mentioned in the file headers
    sinogram_shape = kex_headers.LOR_HISTOGRAM_SHAPE
    number_of_elements = np.prod(sinogram_shape)
    sinograms_3D = []
    paths = []
    
    def sino3d(path):
        with open(path, 'rb') as file:
            sino_1D = np.fromfile(file, count=number_of_elements, dtype=kex_headers.NORM_DTYPE)
        return np.reshape(sino_1D, newshape=sinogram_shape, order='C')
    
    for den in subdirs:
        if only_norm:
            intermediate_file = "norm3d_00.a"
            path = intermediate_folder + r"/" + den + r"/" + intermediate_file
            paths.append(path)
            temp = sino3d(path)
        else:
            intermediate_files = ["emis_00.s", "emis_corr_00.s", "norm3d_00.a"]
            for inter in enumerate(intermediate_files):
                path = intermediate_folder + r"/" + den + r"/" + inter
                paths.append(path)
                temp.append(sino3d(path))
        sinograms_3D.append(temp)
                            
    return sinograms_3D, paths

def reconstructions():
    """ reconstructions: return a list with the KEX data reconstructions, and then also a list with the paths"""
    recon_paths = []
    reconstructions = []
    for b in [False, True]:
        recon_paths +=paths(is_cannula=b)[-1]
        
    for path in recon_paths:
        dicom = pet_images.load_scan(path)
        pixels = pet_images.get_pixels(dicom)
        reconstructions.append(pixels)
    return reconstructions, recon_paths


def norm_components():
    """ norm_components: norm file for KEX dataset (common for cannula and cylinder)
    return matrices, normalization_components, number_of_dimensions, matrix_sizes, matrix_axis_labels"""
    #KEX data
    norm_path = r"C:/Users/petct/Desktop/KanylPET-Converted/KanylPET-norm.n"

    norm_str = "normalization matrix"

    #constants from header
   

    #descriptions
    normalization_components= ["geometric effects",
                              "crystal interference",
                              "crystal efficiencies",
                              "axial effects",
                              "paralyzing ring DT parameters",
                              "non-paralyzing ring DT parameters",
                              "TX crystal DT parameter"]

    #positions in the 1D data file
    data_offset_in_bytes= np.array([0,
                              174400,
                              196800,
                              344640,
                              347124,
                              347344,
                              347564])

    #number of dimensions
    number_of_dimensions=[2,
                          2,
                          2,
                          1,
                          1,
                          1,
                          1]

    #matrix sizes
    matrix_sizes =[(400,109),
                  (14,400),
                  (672,55),
                  621,
                  55,
                  55,
                  14]
    #dimension labels
    matrix_axis_labels =[("sinogram projection bins","sinogram planes"),
                        ("crystal number","sinogram projection bins"),
                        ("crystal number","ring number"),
                        "plane number",
                        "ring number",
                        "ring number",
                        "crystal number"]

    #read data               
    with open(norm_path, 'rb') as file:
        norms_1D=np.fromfile(file, dtype=kex_headers.NORM_DTYPE)

    bytes_per_entry = 4
    last_byte = data_offset_in_bytes[1:]
    last_byte = np.append(last_byte, data_offset_in_bytes[-1]+bytes_per_entry*np.prod(matrix_sizes[-1]))
    data_offset_index= data_offset_in_bytes // bytes_per_entry
    last_index = last_byte // bytes_per_entry

    matrices = []
    for k,dims in enumerate(number_of_dimensions):
        temp = norms_1D[data_offset_index[k]:last_index[k]]
        if (dims>1):
            temp = np.reshape(temp, newshape=matrix_sizes[k], order='F')
        matrices.append(temp)
        
    return matrices, normalization_components, number_of_dimensions, matrix_sizes, matrix_axis_labels

#thank you Dicom tutorial
def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -   
                          slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - 
                      slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

#thank you Dicom tutorial
def get_pixels(scans):
    #get image
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    
    #image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    
    #intercept = scans[0].RescaleIntercept
    #slope = scans[0].RescaleSlope
    
    #if slope != 1:
    #    image = slope * image.astype(np.float64)
    #   image = image.astype(np.int16)
        
    #image += np.int16(intercept)
    return np.array(image, dtype=np.int16)