from copy import deepcopy
from bokeh.core.property.color import Color
import numpy as np
from bokeh import plotting as bk
from bokeh.models import  HoverTool, ColumnDataSource, MultiLine
from matplotlib import colors
from pathlib import Path
import sys
sys.path.append(Path('G:\My Drive\pythonCode\MyPlot'))
# from general4init import *
from .general4init import *

def plotMultipLine(x:np.ndarray, ys:np.ndarray, cmName='myn', showByDefault=False):
    """will plot multiple lines on a single Bokeh plot

    Args:
        x (np.ndarray): a 1D np array
        ys (np.ndarray): 2D numpy array, each row is separate signal
        cmName (str, optional): Name of colormap in str. Defaults to 'myn'.

    Returns:
        [type]: [description]
    """
    N = len(ys)
    p = bk.figure(width=1000, height=600);
    colormap = varyColor(N, usrColormap=cmName)
    # colormap = general4init.varyColor(N, usrColormap=cmName)

    for i in range(N):
        C = colors.to_hex(list(colormap[i]));
        l = p.line(x, ys[i], legend_label="station {}".format(i), line_color=C);
        l.visible=showByDefault
        s = p.square(x, ys[i], legend_label="station {}".format(i), fill_color=None, line_color=C);
        s.visible=showByDefault

    p.legend.click_policy="hide"
    p.add_tools(HoverTool())
    # show(p);
    return p

def coors4QuiverPlot(x,y,u,v, scale=None):

    #--- set auto scaling factor
    if scale is None:
        minDist = np.inf
        import itertools
        for i in itertools.combinations(range(len(x)),2):
            dist = np.sqrt((x[i[0]]-x[i[1]])**2 + (y[i[0]]-y[i[1]])**2)
            if dist<minDist: minDist = deepcopy(dist)
        maxArrowLength = 0.95*minDist
        iMax = np.nanargmax(u**2 + v**2)
        scale = np.sqrt(maxArrowLength/np.sqrt(u[iMax]**2 + v[iMax]**2))
        print('scale determined= {}'.format(scale))
        
        
    xCoors = np.stack((x,x+u*(scale**2)), axis=1)
    yCoors = np.stack((y,y+v*(scale**2)), axis=1)
    X = [list(ii) for ii in xCoors]
    Y = [list(ii) for ii in yCoors]
    return X,Y


def objectsForBokehQuiver(x,y,u,v, scale=None, color='navy', line_width=2):
    """
    Inputs:
        x,y,u,v are 1D vectos shape (N,) containig starting point (x,y)
        and vector length in x,y directions - u,v
    Retrns:
        source: bokeh.models.sources.ColumnarDataSource
        glyph:  bokeh.models.sources.MultiLine
    use outputs of the function as such:
    p.add_glyph(source, glyph)
    where p is a bokeh.plotting.Figure instance
    for example:
        https://docs.bokeh.org/en/latest/docs/gallery/quiver.html
    """

    xCoors, yCoors = coors4QuiverPlot(x,y,u,v, scale)
    source = ColumnDataSource(dict(
        xs=xCoors,
        ys=yCoors,
        )
    )
    glyph = MultiLine(xs="xs", ys="ys", line_color=color, line_width=line_width)
    return source, glyph

def quiver(p, x,y,u,v, scale=None, line_width=2, color='navy'):
    source, glyph = objectsForBokehQuiver(x,y,u,v, scale=scale, color=color, line_width=line_width)
    p.add_glyph(source, glyph)
    p.circle(x,y, size=line_width*2.5, color=color, alpha=0.5)
    return p

if __name__=="__main__":
    from bokeh.io import show
    N = 30
    x = np.sin(np.linspace(1,3,N))
    y = np.tanh(np.linspace(0,3,N))
    u = (np.linspace(0,10,N))**2
    v = (np.linspace(-10,0,N))**2

    import matplotlib.pyplot as plt
    plt.figure()
    plt.quiver(x,y,u,v)
    plt.show()

    p = bk.figure(width=1000, height=600);
    # source, glyph = objectsForBokehQuiver(x,y,u,v)
    # p.add_glyph(source, glyph)
    # p.circle(x,y, size=2.5, color="navy", alpha=0.5)
    p = quiver(p, x,y,u,v)
    show(p)

    pass