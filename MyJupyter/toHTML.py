import numpy as np
import pandas as pd
from copy import deepcopy
from IPython.display import display, HTML
    
def toHTML(obj, **kwargs):
    if type(obj)==dict:
        return dict2HTML(obj, **kwargs)
    elif type(obj)==pd.core.frame.DataFrame:
        return df2HTML(obj, **kwargs)
    elif type(obj)==np.ndarray:
        return ndarray2HTML(obj, **kwargs)
    else:
        return str(obj)


def ndarray2HTML(array:np.ndarray, **kwargs):
    DF = pd.DataFrame(prepareArray4Display(array))
    # DF.style.hide_columns().hide_index()
    return df2HTML(DF, **kwargs)

    pass

def prepareArray4Display(array:np.ndarray):
    array = deepcopy(array)
    array = array.squeeze()
    S = array.shape
    if len(S)==1:
        array = array[None,:]
    elif len(S)>2: 
        arCopy = deepcopy(array)
        s = ''
        for i in range(len(arCopy.shape)-2):
            s += '0,'
            arCopy = arCopy[0]
        array = arCopy
        display(HTML('<font  style="color:#c7370f">Multi dimensional array, displaying the 2D slice: array[{}:,:]</font>'.format(s)))
    return array

def df2HTML(DF:pd.core.frame.DataFrame, \
    style:bool=False, cmap='bwr', low=0, high=0, axis=0, subset=None, text_color_threshold=0.408, vmin=None, vmax=None, gmap=None):

    if style:
        DF_styler = DF.style.background_gradient(
            cmap=cmap,
            low=low,
            high=high,
            axis=axis,
            subset=subset,
            text_color_threshold=text_color_threshold,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap
        )
    else:
        DF_styler = DF
        

    # Getting default html as string
    df_html = DF_styler.to_html() 
    
    # Concatenating to single string
    df_html = CssScrollStyle()+'<div class="dataframe-div">'+df_html+"\n</div>"

    return df_html


def dict2HTML(D, **kwargs):
    header = """
    <table>
    <thead>
        <style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
                }
            tr:nth-child(even) {
                background-color: #a2b0b5;
                color: #444444;
                }
        </style>
    </thead>
    """

    bodyStart = '<tbody>'

    allRows = ''
    for k, v in D.items():
        rowBlock = """
        <tr>
            <th><strong>{}</strong> </th>
            <td style="text-align:left">{}</td>
        </tr>
        """.format(toHTML(k),toHTML(v))
        allRows += rowBlock
    allRows += '</tbody>'

    bodyEnd = """
        </table>
        """

    allHTMLTable = addScrolability2HtmlString(header+bodyStart+allRows+bodyEnd)
    return allHTMLTable


def CssScrollStyle():
    style2Add = """
        <style scoped>
            .dataframe-div {
            max-height: 350px;
            overflow: auto;
            position: relative;
            }

            .dataframe thead th {
            position: -webkit-sticky; /* for Safari */
            position: sticky;
            top: 0;
            background: black;
            color: white;
            }

            .dataframe thead th:first-child {
            left: 0;
            z-index: 1;
            }

            .dataframe tbody tr th:only-of-type {
                    vertical-align: middle;
                }

            .dataframe tbody tr th {
            position: -webkit-sticky; /* for Safari */
            position: sticky;
            left: 0;
            background: black;
            color: white;
            vertical-align: top;
            }
        </style>
        """
    return style2Add

def addScrolability2HtmlString(htmlStr):
    scrollHeader = """
            <div    class="table-container"
                    style="
                        height:320px;
                        overflow:auto;
                        border: 1px outset #d0e0e3;
                        ">
                    """
    scrollEnd = """</div>"""
    return scrollHeader+htmlStr+scrollEnd