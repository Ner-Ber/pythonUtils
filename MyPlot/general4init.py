from numbers import Number
import matplotlib.cm as cm
import matplotlib as mpl
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


if __name__=="__main__":
    plot_color_gradients(4, ['Blues','plasma','myn'])
    plt.show()
    pass