import numpy as np
import scipy as sp
from scipy import io
from pathlib import Path
import matplotlib.pyplot as plt
import sys
# sys.path.append('G://My Drive//pythonCode')
from .. import MyGeneral


def readGracesGeodeticData(path):
    MAT = io.loadmat(str(path))
    relevantKs = [k for k in MAT.keys() if not k.startswith('_')]
    data = {k:createAndExplainDataDict(MAT[k]) for k in relevantKs}   # this assumes a structure with a single element 
    """
    Content should be:
    t - days (sample every 15 sec)
    x,y - polar stereographic projection
    e,n - slip to east and north in meters
    d - slip in stream direction in meters
    z - meters?
    """
    return data

def createAndExplainDataDict(generalTypeArray):
    returnDict = {}
    for n in generalTypeArray.dtype.names:
        returnDict[n] = generalTypeArray[n][0,0]
        print('{}: \t\t {}'.format(n, MyGeneral.whos(returnDict[n], doReturn=True)))
    return returnDict

if __name__=="__main__":
    folderPath = Path('H:\My Drive\icequakes\gps_data')
    fileName = 'WIS11'
    filePath = folderPath / fileName
    data = readGracesGeodeticData(filePath)
    pass