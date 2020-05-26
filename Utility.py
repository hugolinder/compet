# common packages 
import numpy as np 
import os
import struct

from math import *
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

# reading in dicom files
import pydicom

# ipywidgets for some interactive features
from ipywidgets.widgets import * 
import ipywidgets as widgets

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
def get_pixels_hu(scans):
    #get image
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    #print(np.min(image))
    #print(np.max(image))
    
    #image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    
    #intercept = scans[0].RescaleIntercept
    #slope = scans[0].RescaleSlope
    
    #if slope != 1:
    #    image = slope * image.astype(np.float64)
    #   image = image.astype(np.int16)
        
    #image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def offsetMap(size):
    """ the size has to be an odd number"""
    offsets = np.zeros((size,), dtype=int)
    for k in range(0, size // 2):
        offsets[2*k+1] = (k+1)
        offsets[2*k+2] = -(k+1)
    return offsets

#scanner parameters
AXIAL_COMPRESSION = 11
NUMBER_OF_VIEWS = 168
NUMBER_OF_PROJECTIONS = 400
SINOGRAM_SIZE = NUMBER_OF_PROJECTIONS*NUMBER_OF_VIEWS
TOF_OFFSET_MAP = offsetMap(13)
SEGMENT_OFFSET_MAP = offsetMap(9)
SEGMENT_TABLE = np.array([109,97,97,75,75,53,53,31,31], dtype=int)
NUMBER_OF_SINOGRAMS = sum(SEGMENT_TABLE)
MICHELOGRAM_SIZE = SINOGRAM_SIZE*NUMBER_OF_SINOGRAMS
PROJECTION_SPACE_SIZE = MICHELOGRAM_SIZE*(len(TOF_OFFSET_MAP)+1)
#calculate "function" from michelogram index to segment and axial offset
SEGMENT_INDEX_MAP = np.zeros((NUMBER_OF_SINOGRAMS,), dtype=np.int)
""" precalculated function from michelogram index to segment index """
HISTOGRAM_SHAPE=(1+len(TOF_OFFSET_MAP),
                 NUMBER_OF_SINOGRAMS,
                 NUMBER_OF_VIEWS,
                 NUMBER_OF_PROJECTIONS)
HISTOGRAM_DIMENSION_TITLES=["TOF bin ", "michelogram bin ", "transaxial angle bin ", "radial offset bin "]
IMAGE_SHAPE = (SEGMENT_TABLE[0],
              NUMBER_OF_PROJECTIONS //2,
              NUMBER_OF_PROJECTIONS //2)
IMAGE_DIMENSION_TITLES = ["z pixel (gantry to bed)", "y pixel (up to down)", "x pixel (left to right)"]
radians_per_angle_bin = np.pi /NUMBER_OF_VIEWS
phi_radians = np.arange(NUMBER_OF_VIEWS)*radians_per_angle_bin
TX_ANGLE_TO_RX = np.cos(phi_radians)
TX_ANGLE_TO_RY = np.sin(phi_radians)

AXIAL_INDEX_MAP = np.zeros(SEGMENT_INDEX_MAP.shape, dtype=np.int)
""" precalculated function from michelogram index to axial position [halfplane] """
m = 0
for s in range(0,len(SEGMENT_TABLE)):
    SEGMENT_INDEX_MAP[m:m+SEGMENT_TABLE[s]] = s
    AXIAL_INDEX_MAP[m:m+SEGMENT_TABLE[s]] = np.arange(0,SEGMENT_TABLE[s], 1) + (SEGMENT_TABLE[0] - SEGMENT_TABLE[s])//2
    m += SEGMENT_TABLE[s]

def mi_from_seg(segment_index):
    """ returns numpy array with michelogram indices (0,1, ...) in the specified segment (0,1,2, ...)"""
    mi_low = np.sum(SEGMENT_TABLE[:segment_index])
    return mi_low + np.arange(SEGMENT_TABLE[segment_index])

def mi_bounds_from_seg(segment_index):
    """ returns minimum, maximum michelogram indices in the specified segment (0,1,2, ...)"""
    mi_low = np.sum(SEGMENT_TABLE[:segment_index])
    mi_high = mi_low+SEGMENT_TABLE[segment_index]
    return mi_low, mi_high

def mi_from_seg_ao(segment_index, axial_index):
    """ returns michelogram index (0,1, ...) for segment index (0,1,2,...) and axial index (halfplane number 0,1, ...)"""
    mi = np.sum(SEGMENT_TABLE[:segment_index])
    mi += axial_index - (SEGMENT_TABLE[0] - SEGMENT_TABLE[segment_index])//2
    return mi

#a faster way of CenterOfMass (compared to recursive version )
def CenterOfMassMultiplication(arr, axis=None):
    """calculate Center of mass of numpy array with array multiplication using array coordinates. \n
    Center of mass is calculated for dimensions specified with axis, allowing vectorization. \n
    return: coordinates / counts (floats) """
    #calculate 1-D representations of grid
    
    arr64 = np.array(arr, dtype=np.int64)
    if axis is None:
        axis = tuple(np.arange(len(arr64.shape)))
    counts = np.sum(arr64, axis=axis)
    coordinate_shape = [len(axis)]
    for k in range(0,len(arr64.shape)):
        if k not in axis:
            coordinate_shape.append(arr64.shape[k])
    #print(coordinate_shape)
    
    coordinates = np.zeros(shape=tuple(coordinate_shape), dtype=np.int64)
    for k in range(0,len(axis)): 
        position = np.arange(0,arr64.shape[axis[k]],1)
        position_shape = np.ones(len(arr64.shape), dtype=np.int64)
        position_shape[axis[k]] = arr64.shape[axis[k]]
        position = np.reshape(position, tuple(position_shape))
        weighted_positions = arr64*position
        flatten = np.sum(weighted_positions, axis=axis)
        coordinates[k] = flatten
    return coordinates/counts

#thank you GeeksforGeeks
def bitExtracted(number, k, p): 
    """ extract k bits towards most significant bit, starting at p bits away from least significant bit (where p=1) """
    # k 1's: 111111...111 AND the number starting at bit index (p-1) [0 indexing]
    return ( ((1 << k) - 1)  &  (number >> (p-1) ) );

#Thank you Siemens 4 ring mCT walkthrough
def binAddresToProjectionSpace(event):
    """ extract [tofbin,mi,tx,robin] bin using formula: event = 1*robin+c1*tx+c1*c2*mi+c1*c2*c3*tof + prompt + tag 
   robin = ro + 200"""
    # tag = bitExtracted(event,1,32)
    #prompt_bit = bitExtracted(event,1,31)
    event = np.array(event)
    bin_address_remains = event & 0X3FFFFFFF
    coefficients = [MICHELOGRAM_SIZE, SINOGRAM_SIZE, NUMBER_OF_PROJECTIONS, 1 ]
    coordinates = np.zeros((len(coefficients),np.size(event)), dtype=int)
    for k in range(0, len(coefficients)):
        coeff = coefficients[k]
        coordinates[k] = bin_address_remains // coeff
        bin_address_remains -= coordinates[k]*coeff
    #coordinates[-1] -= NUMBER_OF_PROJECTIONS // 2 #reintroduce negative range
    #coordinates = np.flip(coordinates, axis=0)
    coordinates = np.squeeze(coordinates)
    return coordinates

def projectionSpaceToBinAddres(coordinates):
    """ convert coordinates [ro, tx, mi, tof] to 1D bin address"""
    temp = np.multiply(coordinates, [1,
                                        NUMBER_OF_PROJECTIONS, 
                                        SINOGRAM_SIZE, 
                                        MICHELOGRAM_SIZE])
    s = np.sum(temp)
    return int(s + NUMBER_OF_PROJECTIONS//2)

def isEvent(event):
    tag = (event >> 31) & 1
    if (tag == 0): #event tag
        return True
        # prompt_bit = (event >> 30) & 1
        # if (prompt_bit== 0): #delayed
        # else: # coincidence
    else: #packet
        #if (((event >> 28) & 0Xe) == 8): #elapsed time tag 
        #if (( event >> 24 & 0XFF) == 0XC4): #horizontal bed position 
        #if ((event >> 24 & 0XFC) == 0XBC): #lost event tally
        return False
    
def timeSlices(dataLM, timeTagsPerSlice=1):
    """ split Listmode data according to time tags """
    bTagAndTime = np.zeros((2, len(dataLM)), dtype=np.bool)
    bTagAndTime[0] = ((dataLM >> 31) & 1) == 1
    bTagAndTime[1] = ((dataLM >> 28) & 0Xe) == 8
    timeTags = np.nonzero(np.all(bTagAndTime, axis=0))[0]
    
    splits = timeTags[timeTagsPerSlice: :timeTagsPerSlice]
    return np.split(dataLM, splits)

def events(dataLM):
    """ takes numpy array of 32 bit listmode data and only returns the events (that is, tags removed)"""
    return dataLM[((dataLM >> 31) & 1) == 0]

def delaysOfEvents(events):
    """ takes numpy array of 32 bit listmode data with events and only returns the delays (that is, prompts removed)"""
    return events[(events >> 30) & 1 == 0]

def bDelays(events):
    """ a bool array indicating delays in LM events"""
    return (events >> 30) & 1 == 0

def promptsOfEvents(events):
    """ takes numpy array of 32 bit listmode data with events and only returns the prompts (that is, delays removed)"""
    return events[(events >> 30) & 1 == 1]

def bPrompts(events):
    """ a bool array indicating prompts in LM events"""
    return (events >> 30) & 1 == 1

def binAddressFromLM(LM):
    """ extract the bin adress bits of LM events """
    return LM & 0X3FFFFFFF 
    
def get2DMichelogramIndex(michelogram_index):
    """ converts 1D michelogram index to segment number (tilt) and axial offset (halfplanes). \n precalculated segment_index_map and axial_index_map in Utility are faster """
    #mi / seg_table[0]
    # 55 + 55 -1 
    segment_bins = np.cumsum(SEGMENT_TABLE)
    segment_number = np.searchsorted(segment_bins, michelogram_index+1, side='left', sorter=None)
    
    
    if segment_number == 0:
        axial_offset = michelogram_index 
    else:
        segment_index = michelogram_index - segment_bins[segment_number-1]
        axial_offset = segment_index + (SEGMENT_TABLE[0] - SEGMENT_TABLE[segment_number])//2 
    
    return segment_number, axial_offset

def binaryString(number):
    return "{0:b}".format(number)

def listmodePathToHistogram(listmode_path):
    """ reads listmode data from listmode_path and returns a 4D numpy array with the histogram. \n
    the dimensions are transaxial angle, radial offset, michelogram index, and tof bin"""
    #read
    with open(listmode_path, 'rb') as file:
        listmode_data = np.fromfile(file, dtype=np.uint32)
    #Extract event data
    event_data = listmode_data[((listmode_data >> 31) & 1) == 0]
    #extract position in histogram
    bin_addresses=event_data & 0X3FFFFFFF 
    #construct histogram
    histogram = np.zeros((PROJECTION_SPACE_SIZE,),dtype=np.ushort)
    np.add.at(histogram, bin_addresses,1) #histogram[bin_addresses] += 1 # does not work as expected
    
    return np.reshape(histogram, HISTOGRAM_SHAPE)

def viewNDImages(images, fixedDimensions, fixedValues,  subplotShape = None, 
                 titles=None, axisLabels=None, figsize=None):
    """ plots 2D slices of N-dimensional images in subplots at position fixedValue of image dimensions fixedDimensions. """
    #print("fixD", fixedDimensions)
    #print("fixV", fixedValues)
    #print("axis labels", axisLabels)
    
    #try to find a nice way to grid them
    #assume images have the same size
    images = np.array(images)
    #make sure the array is sorted
    index_array = np.argsort(fixedDimensions)
    #print("index ar", (np.array(index_array),))
    fixedDimensions = np.array(fixedDimensions)[index_array]
    fixedValues = np.array(fixedValues)[index_array]
    #change axis labels correspondingly
    if axisLabels is None:
        axisLabels = [None, None]
    
    #determine which axes to move to the back, per image
    freeDimensions = np.ones((images.ndim-1,), dtype=int)
    freeDimensions[list(fixedDimensions)] = 0
    source = np.nonzero(freeDimensions)[0] + 1
    destination = (-2,-1)
    
    # " Move axes of an array to new positions.
    # Other axes remain in their original order. "
    images = np.moveaxis(a=images, source=source,destination=destination)
    plt.figure(figsize=figsize)
    if subplotShape is None:
        subplotShape = [len(images), 1]
    for k in range(len(images)):
        plt.subplot(subplotShape[0], subplotShape[1], k+1)
        plt.imshow(images[(k,) + tuple(fixedValues)])
        plt.xlabel(axisLabels[1])
        plt.ylabel(axisLabels[0])
        if titles is not None:
            plt.title(titles[k])
            
def interactHistograms(histograms, titles=None, subplotShape=None, figsize=None):
    """ assumes a sequence of 4D histograms [tof, mi, tx, ro] """
    #determine view
    #slice dimension 1
    #slice dimension 2
    dimensions = np.arange(4)
    horizontalWidget = widgets.IntSlider(min=0,max=4-1, value=3,
                                         description='horiz. axis')
    verticalWidget = widgets.IntSlider(min=0, max=4-1, value=2, description = 'vert. axis')
    #check difference with an if statement
    
    #determine position on fixed dimensions
    # for example mi, tof
    
    positionWidgetA = widgets.IntSlider(min=0)
    positionWidgetB = widgets.IntSlider(min=0)
    
    #update
    def fixedDimensions(horiz, vert):
        dimensions = np.arange(4)
        #print("bFixed2D", [dimensions != horiz, dimensions != vert])
        #print("bFixed", np.all([dimensions != horiz, dimensions != vert], axis=0))
        bFixedDims = np.all([dimensions != horiz, dimensions != vert], axis=0)
        fixedDims = dimensions[bFixedDims]
        return fixedDims
    
    def updatePositionWidgets(*args):
        fixedDims = fixedDimensions(verticalWidget.value, horizontalWidget.value)
        for k, widg in enumerate([positionWidgetA, positionWidgetB]):
            dim = fixedDims[k]
            widg.max = HISTOGRAM_SHAPE[dim] -1
            widg.description = HISTOGRAM_DIMENSION_TITLES[dim]
            
    for widg in [verticalWidget, horizontalWidget]:
        widg.observe(updatePositionWidgets, 'value')
    
    def f(vDim, hDim, posA,posB):
        if vDim != hDim:
            fixedDims =fixedDimensions(vDim, hDim)
            #print(fixedDims)
            viewNDImages(images=histograms, fixedDimensions=fixedDims, fixedValues=[posA, posB],
                                subplotShape=subplotShape, titles = titles,
                                axisLabels=[HISTOGRAM_DIMENSION_TITLES[hDim],HISTOGRAM_DIMENSION_TITLES[vDim]],
                                            figsize=figsize)
        else: 
            print("warning: non-distinct vertical and horizontal axes")
    
    interact(f, vDim=verticalWidget, hDim=horizontalWidget, posA=positionWidgetA, posB=positionWidgetB);
    
def interactImages(images, titles=None, subplotShape=None, figsize=None):
    """ assumes a sequence of 3D images [z,y,x] """
    #determine view
    #fixed dimension
    dimensions = np.arange(3)
    fixedDimensionWidget = widgets.IntSlider(min=0, max=dimensions[-1], value=0)
    
    #determine position on fixed dimension
    # for example z
    
    positionWidget = widgets.IntSlider(min=0)
    
    #updates
    def updatePositionWidget(*args):
        positionWidget.max = IMAGE_SHAPE[fixedDimensionWidget.value] -1
        positionWidget.description = IMAGE_DIMENSION_TITLES[fixedDimensionWidget.value]
            
    fixedDimensionWidget.observe(updatePositionWidget, 'value')
    updatePositionWidget()
    
    def f(fixDim, fixPos):
            #print(fixedDims)
            #determine image axes from fixdim
            dimensions = np.arange(3)
            hDim,vDim = dimensions[dimensions != fixDim]
            
            viewNDImages(images=images, fixedDimensions=[fixDim], fixedValues=[fixPos],
                                subplotShape=subplotShape, titles = titles,
                                axisLabels=[IMAGE_DIMENSION_TITLES[hDim],IMAGE_DIMENSION_TITLES[vDim]],
                                            figsize=figsize)
    
    interact(f, fixDim=fixedDimensionWidget, fixPos=positionWidget);
    

    
def bPointInFOV(point3D):
    for k, pos in enumerate(point3D):
        if pos < 0:
            return False
        if pos > IMAGE_SHAPE[k]-1:
            return False
    return True