import numpy as np
import pydicom
import os

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