import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from .toHTML import toHTML



def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def what_startup():
    if isnotebook():
        print('notebook start')
    else:
        print('terminal strat')


def display_jupy(obj, **kwargs):
    htmlObj = toHTML(obj, **kwargs)
    display(HTML(htmlObj))



def display_dict(Dict, sortListVals=True):
    return display(HTML(toHTML.dict_2_html_table(Dict, sortListVals=sortListVals)))


def set_jupyter_display():
    get_ipython().magic(u"%matplotlib inline")
    # get_ipython().magic(u"%matplotlib notebook")
    plt.style.use('notebook')

    #--- other notbook configs:
    get_ipython().magic(u"%load_ext autoreload")
    get_ipython().magic(u"%autoreload 2")




if __name__=='__main__':
    print(isnotebook())
