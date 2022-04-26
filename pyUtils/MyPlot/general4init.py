from numbers import Number
from threading import local
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
   

def toHTML(obj):
    if type(obj)==dict:
        pass
    elif type(obj)==dict:
        pass


def createColormap(vmin_vmax=None, cmap=cm.rainbow):
    if vmin_vmax is None:
        vmin_vmax = (0,1)
    elif type(vmin_vmax)==np.ndarray:
        vmin_vmax = (vmin_vmax.ravel().min(), vmin_vmax.ravel().max())

    norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def number2colors(numbers, vmin_vmax=None, cmap=cm.rainbow):
    """take a vector of numbers and translate them into and array where each row is a 4-digit rgba color

    Args:
        numbers (_type_): input array of num
        vmin_vmax (tuple of 2 nums, optional): range of numbers to use. Defaults to None.
        cmap (matplotlib colormap or colormap name): the colormap to create from. Defaults to cm.rainbow.

    Returns:
        _type_: _description_
    """
    if type(cmap)==str:
        cmap = getattr(cm, cmap)
    if vmin_vmax is not None:
        vmin = np.minimum(vmin_vmax)
        vmax = np.maximum(vmin_vmax)
        numbers = np.maximum(np.minimum(numbers, vmax), vmin)
    m = createColormap(vmin_vmax=numbers, cmap=cmap)
    return m.to_rgba(numbers)


def varyColor(NumberOfSets, usrColormap='myn'):
    """create an array where each row is a color sampled from a colormap.
       Number of rows is determined by user.

    Args:
        NumberOfSets (int): number of colors in resulting color array
        usrColormap (str, optional): Name of a pyplot colormap or 'myn'. Defaults to 'myn'.

    Returns:
        np.ndarray: array of size NumberOfSets*3 (number of colors by R,G,B)
    """

    if usrColormap.lower()=='myn':
        tempCmap = myColorMap(NumberOfSets)
    else:
        cmInst = cm.__getattribute__(usrColormap)
        tempCmap = cmInst(range(256))
    
    return  interpColorMap(tempCmap, NumberOfSets)

def myColorMap(NumberOfSets):
    """return a nice color map I created

    Args:
        NumberOfSets (int): how many objects you'll need to plot
    """


    Colors = [
        [213,62,79],
        [244,109,67],
        [253,174,97],
        [254,224,139],
        [255,255,191],
        [230,245,152],
        [171,221,164],
        [102,194,165],
        [50,136,189]
    ]

    return np.array(Colors)/255


def interpColorMap(cmap, num2interps2):
    x = np.linspace(0,1,len(cmap))
    f = interpolate.interp1d(x, cmap, axis=0)
    xnew = np.linspace(0,1,num2interps2)
    return f(xnew)


def plot_color_gradients(numOfItems, cmap_list=None):
    # Create figure and adjust figure height to number of colormaps
    if cmap_list is None:
        cmap_list = ['plasma']
    elif type(cmap_list)==str:
        cmap_list = [cmap_list]
    nrows = len(cmap_list)

    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)

    for ax, name in zip(axs, cmap_list):
        gradient = varyColor(numOfItems, name)
        gradient = np.reshape(gradient[:,:3],(1,-1,3))

        # ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.imshow(gradient, aspect='auto')
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

def colorline(
    x, y, z=None, ax=None, cmap=plt.get_cmap('jet'), norm=None,
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    if norm is None:
        norm = plt.Normalize(z.min(), z.max())
    else:
        assert len(norm)==2 # should be tuple/list of two scalars
        norm = plt.Normalize(*norm)


    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale(axis='both')
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


if __name__=="__main__":
    plot_color_gradients(4, ['Blues','plasma','myn'])
    plt.show()
    pass
