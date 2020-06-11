import matplotlib.pyplot as plt
import numpy as np

def get_drill_crack(ndarray, position, cracks):
    """ analogy: drill to position in ndarray and get data for all indices through the cracks, that is the dimension indices"""
    pos = position.copy()
    for crack in cracks:
        pos[crack] = slice(ndarray.shape[crack])
    return ndarray[tuple(pos)]

def plot_views(image, 
               position, 
               viewpairs=None, 
               axes=None, 
               dimlabels=None,
               img_title=None,
               **imshowkwargs): 
    """ imshow 2d views of image, with axes determined by viewpair"""
    if viewpairs is None:
        viewpairs = get_all_viewpairs(image.ndim)
    if axes is None:
        _, axes = plt.subplots(ncols=len(viewpairs))
    ims = []
    for ax, viewpair in zip(axes, viewpairs):
        if len(imshowkwargs) > 0:
            print("imshowkwargs", imshowkwargs)
        ims.append(ax.imshow(get_drill_crack(image, position, viewpair), **imshowkwargs))
        pos_str = get_position_string(position, viewpair)
        if dimlabels is None:
            pos_str = "[ ]".replace(" ", pos_str)
        else:
            context = get_position_string(dimlabels, [])
            pos_str = " ".join([", at", context, "=", pos_str])
        if img_title is None:
            img_title = "view"
        ax.set_title(img_title + pos_str)
    set_labels(viewpairs, axes, dimlabels)
    return ims

def set_labels(viewpairs, axes, dimlabels=None):
    """ labels the axes for plot views"""
    if dimlabels is None:
        ndims = max([max(pair) for pair in viewpairs]) + 1 
        dimlabels = ["axis {}".format(d) for d in range(ndims)]
    xlabels = [dimlabels[max(pair)] for pair in viewpairs]
    ylabels = [dimlabels[min(pair)] for pair in viewpairs]
    for k, ax in enumerate(axes):
        ax.set_xlabel(xlabels[k])
        ax.set_ylabel(ylabels[k])

def get_position_strings(position, viewpairs):
    """ framed '(x, y, z )' coordinates, with fixes position replaced with ':' at viewpair indices """
    strings = []
    for pair in viewpairs:
        position_str = [str(k) for k in position]
        for p in pair:
            position_str[p] = ":"
        strings.append("( )".replace(" ", ", ".join(position_str)))
    return strings

def get_position_string(position, freedim):
    """ framed '(x, y, z )' coordinates, with fixes position replaced with ':' at viewpair indices """
    position_str = [str(k) for k in position]
    for p in freedim:
        position_str[p] = ":"
    return "( )".replace(" ", ", ".join(position_str))

def get_all_viewpairs(ndims):
    viewpairs = []
    for a in range(ndims):
        for b in range(a+1,ndims):
            viewpairs.append((a,b))
    return viewpairs

def setup_fig_axes(images, scale=10):
    """ return fig, axes from plt.subplots to use with images"""
    vert = len(images)
    horz = images[0].ndim
    shape = (horz*scale, vert*scale)
    return plt.subplots(nrows=vert, ncols=horz, figsize=shape)


