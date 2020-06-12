import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker

#Interactive
from ipywidgets.widgets import * 
import ipywidgets as widgets

#Hugo code
import kex_headers

def same_colorbar(fig, images, axes, nbins=5, clim=None):
    """ make a shared colorbar that applies to the images on the axes. 
    The colorbar has up to nbins ticks"""
    if clim is None:
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
    else:
        vmin, vmax = clim
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images: 
        im.set_norm(norm)
    cb = fig.colorbar(images[0], ax=axes)
    tick_locator = ticker.MaxNLocator(nbins=nbins)
    cb.locator = tick_locator
    cb.update_ticks()

def plot_views(image, viewpairs, position, axes):    
    for ax, viewpair in zip(axes, viewpairs):
        index = position.copy()
        for k in range(2):
            dim = viewpair[k]
            index[dim] = slice(image.shape[dim])
        index = tuple(index)
        ax.imshow(image[index])

def volume_views(image, position, axes):
    ndims = len(position)
    views = []
    for dim, pos,ax in zip(range(ndims), position, axes):
        view = np.swapaxes(image, 0, dim)[pos]
        #print("view", view)
        views.append(ax.imshow(view))

    return views

def plot_volume_views(volume_dict, dimlabels, position, **kwargs):
    fig, axes = plt.subplots(nrows=len(image_dict), 
                             ncols=len(dimlabels), 
                             **kwargs)
    rows = dict(zip(image_dict.keys(), axes))
    for dkey, img in image_dict.items():
        row = rows[dkey]

        views = volume_views(img, position, row)
        same_colorbar(fig, views, row)
        label_volume_views(dimlabels, position, row)
        title_volume_views(dkey, 
                                             position, 
                                             row, 
                                             dimlabels)

    
def title_volume_views(title, position, axes, dimlabels=None):
    ndims = len(position)
    if dimlabels is None: 
        dimlabels = ["dim {}".format(k) for k in range(ndims)]
    for k, diml, pos, ax in zip(range(ndims), dimlabels, position, axes):
        ax.set_title(title + ", " + diml + "={}".format(pos))

def label_volume_views(dimlabels, position, axes):
    for k, ax in enumerate(axes):
        temp = dimlabels.copy()
        temp[0] = dimlabels[k]
        temp[k] = dimlabels[0]
        ax.set_xlabel(temp[-1])
        ax.set_ylabel(temp[-2])
      

def plot_com_trace(com_trace, 
                   hvalues=None,
                   space_shape=None, 
                   hlabel=None, 
                   legends=None, 
                   dim_str=None,
                   hlines=None, 
                   hlines_fromto=None,
                   vlines=None, 
                   title=None,
                   marker=None,
                   sharex=False,
                   std=None,
                   **fig_kw):
    if space_shape is None:
        reset_ylim=[False]
    else:
        reset_ylim = [False,True]
    columns = len(reset_ylim)
    ndims = len(com_trace)
    if dim_str is None:
        dim_str = []
        for d in range(ndims):
            dim_str.append("center of mass\ndim. " + str(d))
    if legends is None:
        legends = ["trace", "ref.", "gates"]
    if hvalues is None: hvalues = np.arange(len(com_trace[0]))
    fig, axes = plt.subplots(nrows=ndims,ncols=columns, sharex=sharex, **fig_kw)
    for dim_number in range(ndims):
        for k, redo_y in enumerate(reset_ylim): 
            ax = axes[dim_number,k]
            x = hvalues
            y = com_trace[dim_number]
            di = {"marker":marker, 'label':legends[0], 'zorder':1}

            if std is None:
                ax.scatter(x,y,**di)
            else:
                #ax.errorbar(hvalues, com_trace[dim_number], yerr=std,marker=marker, label= zorder=1)
                ax.errorbar(x,y,yerr=std, **di)
            if redo_y:
                ax.set_ylim(0, space_shape[dim_number])
            if hlines is not None:
                if hlines_fromto is not None:
                    ha = hlines_fromto[0]
                    hb = hlines_fromto[1]
                else:
                    ha = hvalues[0]
                    hb = hvalues[-1]
                ax.hlines(hlines[dim_number], ha, hb, linestyles='dashed', label=legends[1], zorder=3)
            if vlines is not None:
                ax.vlines(vlines, ax.get_ylim()[0],ax.get_ylim()[-1], linestyles='dotted', label=legends[2], zorder=3)
            ax.set_ylabel(dim_str[dim_number], fontsize='x-large')
            if not sharex:
                ax.set_xlabel(hlabel, fontsize='x-large')
            elif dim_number == ndims-1: 
                ax.set_xlabel(hlabel, fontsize='x-large')
            ax.legend(loc='lower right')
    plt.suptitle(title)

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
        positionWidget.max = pet_scan.IMAGE_SHAPE[fixedDimensionWidget.value] -1
        positionWidget.description = pet_scan.IMAGE_DIMENSION_TITLES[fixedDimensionWidget.value]
            
    fixedDimensionWidget.observe(updatePositionWidget, 'value')
    updatePositionWidget()
    
    def f(fixDim, fixPos):
            #print(fixedDims)
            #determine image axes from fixdim
            dimensions = np.arange(3)
            hDim,vDim = dimensions[dimensions != fixDim]
            
            viewNDImages(images=images, fixedDimensions=[fixDim], fixedValues=[fixPos],
                                subplotShape=subplotShape, titles = titles,
                                axisLabels=[pet_scan.IMAGE_DIMENSION_TITLES[hDim],pet_scan.IMAGE_DIMENSION_TITLES[vDim]],
                                            figsize=figsize)
    
    interact(f, fixDim=fixedDimensionWidget, fixPos=positionWidget);