from copy import deepcopy
import os
import re
import time
from pathlib import Path, PurePath
import json
import requests
import pandas as pd
from scipy import signal
import numpy as np
# import lxml
from obspy.clients.fdsn import Client, header
import obspy
from obspy.core import UTCDateTime
from obspy.core.event import Catalog
import sys
# sys.path.append('G:\\My Drive\\pythonCode')
from pyUtils import MyGeneral
from  obspy.core.event.catalog import read_events


def japanHypocenterRecordFormat():
    # html = requests.get(url).content
    htmlFile=Path("G:\My Drive\PhD\earthquakeData\japanHypeCenterFormat.html")
    df_list = pd.read_html(htmlFile)
    df = df_list[-1]
    print(df)
    df.to_csv('my data.csv')



def getEventsViaObspy(clientName='NCEDC', **kwargs):
    """
    a wrapper function to get events using obspy API.
    To see options in kwargs go here:
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_events.html
    """
    
    #-- set the clinet
    # client = Client(clientName, timeout=3600)
    client = Client(clientName, timeout=1)

    #--- set defaults for the get_events
    startTime=obspy.UTCDateTime(2020,1,1)
    endTime=startTime + 60*60*24*7
    kwargs.setdefault('starttime',  startTime)
    kwargs.setdefault('endtime',    endTime)
    # kwargs.setdefault('limit',    int(1e15))

    catalog = client.get_events(**kwargs)
    return catalog


def getEventsViaObspy_catchException(clientName='NCEDC', **kwargs):
    """
    a wrapper function to get events using obspy API.
    As opposed to getEventsViaObspy this function will catch the timeout exception,
    split the time window into smaller bits and retry to get the catalog.
    """
    
    #-- set the clinet
    client = Client(clientName, timeout=30)

    #--- set defaults for the get_events
    startTime=obspy.UTCDateTime(2020,1,1)
    endTime=startTime + 60*60*24*7
    kwargs.setdefault('starttime',  startTime)
    kwargs.setdefault('endtime',    endTime)

    catalog = getEventsViaObspy_setTime_recurse(client, kwargs['starttime'], kwargs['endtime'], N_split=5, **kwargs)
    return catalog



def getEventsViaObspy_setTime_recurse(client, startTime, endTime, N_split=5, **input_params):

    try:
        catalog = client.get_events(**input_params)
    except obspy.clients.fdsn.header.FDSNNoDataException as nd:
        print(f"not data, exception caught: {nd}")
        catalog = Catalog() # return an empty catalog
    except obspy.clients.fdsn.header.FDSNException as to:
        print(f'Timeout (probably).  \\\
            \nexception caught: {to} \\\
            \nSplitting requested time to smaller slots')
        
        #--- will retry for shorter timeperiods
        #-- the requested time limits:
        input_params_mod = deepcopy(input_params)
        input_params_mod['starttime'] = startTime 
        input_params_mod['endtime'] = endTime 
        catalog = Catalog() # create an empty catalog
        dt_vec = np.linspace(startTime.timestamp, endTime.timestamp, N_split) # split requested timeperiod into shorter timeperiod
        #-- iterate upon split timeperiods
        for i in range(N_split-1):
            start_new = UTCDateTime(dt_vec[i])
            end_new = UTCDateTime(dt_vec[i+1])
            cat_part = getEventsViaObspy_setTime_recurse(client, start_new, end_new, N_split=N_split, **input_params_mod)
            catalog.extend(cat_part)
    return catalog

def obspyCat2Df(catalog):
    """will take a catalog generated by getEventsViaObspy and trans it to pandas DF

    Args:
        catalog (obspy catalog): [can be generated by getEventsViaObspy]

    Returns:
        [pandas DF]: [selected fields]
    """
    catalogDf = pd.DataFrame()
    for E in catalog.events:
        D_origin = obspyOrigin2Dict(E.origins[0])       ## TODO: this should be temporary. why are there multiple origins? should I really take the first one only?
        # create a list of dictionaries, one per magnitude listing
        listOfMagDicts = [obspyMagnitude2Dict(mag) for mag in E.magnitudes] if len(E.magnitudes)>0 else [obspyMagnitude2Dict(None)]
        # the dataframes
        headerList = list(D_origin.keys()) + list(listOfMagDicts[0].keys())
        catDf = pd.DataFrame(columns=headerList)
        for D in listOfMagDicts:
            catDf = catDf.append({**D_origin, **D}, ignore_index=True)
        catalogDf = catalogDf.append(catDf, ignore_index=True)
    return catalogDf

def obspyCat2File(catalog, filePath):
    """similar to obspyCat2Df but will write into  afile 

    Args:
        catalog (obspy catalog): [can be generated by getEventsViaObspy]
        filePath ([str or Path]): [where to write the catalog to, will overwrite if already existing]
    """
    writeHeader = True  # write header only on the first writing
    counter = 0
    with open(filePath, 'w') as fileHandle:
        for E in catalog.events:
            D_origin = obspyOrigin2Dict(E.origins[0])       ## TODO: this should be temporary. why are there multiple origins? should I really take the first one only?
            # create a list of dictionaries, one per magnitude listing
            listOfMagDicts = [obspyMagnitude2Dict(mag) for mag in E.magnitudes] if len(E.magnitudes)>0 else [obspyMagnitude2Dict(None)]
            # write data headers:
            headerList = list(D_origin.keys()) + list(listOfMagDicts[0].keys())
            if writeHeader:
                headerStr = ' '.join(headerList) + '\n'
                fileHandle.write(headerStr)
                writeHeader=False
            
            catDf = pd.DataFrame(columns=headerList)
            for D in listOfMagDicts:
                catDf = catDf.append({**D_origin, **D}, ignore_index=True)
            catDf = catDf.reindex(headerList, axis=1)
            dfAsStr = catDf.to_string(header=False, index=False) + '\n'
            fileHandle.write(dfAsStr)
            # catDf.to_csv(fileHandle, sep=' ', index=False, header=writeHeader, mode='a')
            writeHeader = False
            counter+=1
    return

def obspyOrigin2Dict(Origin):
    D = UTCDateTime2Dict(Origin.time)
    D['longitude'] = Origin.longitude
    D['longitude_err'] = Origin.longitude_errors['uncertainty']
    D['latitude'] = Origin.latitude
    D['latitude_err'] = Origin.latitude_errors['uncertainty']
    D['depth'] = Origin.depth
    D['depth_err'] = Origin.depth_errors['uncertainty']
    return D

def obspyMagnitude2Dict(Magnitude=None):
    D = {}
    if Magnitude is not None:
        D['mag'] = Magnitude.mag
        D['mag_error'] = Magnitude.mag_errors['uncertainty']
        D['magnitude_type'] = Magnitude.magnitude_type
    else:
        D['mag'] = D['mag_error'] = D['magnitude_type'] = None
    return D

def UTCDateTime2Dict(UTCDateTime):
    D = {}
    D['year'] = UTCDateTime.year
    D['month'] = UTCDateTime.month
    D['day'] = UTCDateTime.day
    D['hour'] = UTCDateTime.hour
    D['minute'] = UTCDateTime.minute
    D['second'] = UTCDateTime.second
    return D

def kwargsDefaults(kwargs:dict):
    """return default kwargs for the get_events obspy function
        for elaboration about these arguments:
        https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_events.html
    Args:
        kwargs (dict): the kwargs that contain parameters for the obspy function

    Returns:
        kwargs with defaults
    """
    kwargs = deepcopy(kwargs)
    #--- set defaults for the get_events
    startTime=obspy.UTCDateTime(2020,1,1)
    endTime=startTime + 60*60*24*100
    kwargs.setdefault('starttime',  startTime)
    kwargs.setdefault('endtime',    endTime)
    kwargs.setdefault('latitude',  35.07)
    kwargs.setdefault('longitude',  -119.39)
    kwargs.setdefault('maxradius',  2)
    kwargs.setdefault('mindepth',  0)
    kwargs.setdefault('maxdepth',  6000)
    kwargs.setdefault('minmagnitude',  0)
    kwargs.setdefault('maxmagnitude',  20)
    kwargs.setdefault('magnitudetype',  None)
    return kwargs


def createAndSaveCat(containingFolderPath, clientName='NCEDC', saveAsTxt=False, **kwargs):
    """cretae a txt catalog according to your specifications.

    Args:
        containingFolderPath (str, or Path): location to save data. Will create a folder by this name cntaining all relevant data
        client (str, optional): [description]. Defaults to 'NCEDC'.
        
    Default Keyword Argument are set in the function kwargsDefaults:
    """
    #--- set defaults for the get_events
    kwargs = kwargsDefaults(kwargs)

    #-- create a folder for saving cat and readme
    createdFolderPath = createFolderForSavingCat(containingFolderPath, clientName)
    os.mkdir(createdFolderPath);

    # cat = getEventsViaObspy(clientName=clientName, **kwargs)
    cat = getEventsViaObspy_catchException(clientName=clientName, **kwargs)

    #-- save catalog
    if saveAsTxt:
        filePath = createdFolderPath / 'cat.txt'
        obspyCat2File(cat, filePath)
    else:
        filePath = createdFolderPath / 'cat.xml'
        cat.write(filePath, format="QUAKEML")  

    #-- create a readme file from kwargs
    readmePath = createdFolderPath / 'catDetails.txt'
    saveReadmeFileFromKwargs(readmePath, kwargs, clientName)



def saveReadmeFileFromKwargs(readmePath, kwargs, clientName):
    kwargs = deepcopy(kwargs)
    #-- create a readme file from kwargs
    T = time.localtime()
    Tstr = str(T.tm_year)+'_'+str(T.tm_mon)+'_'+str(T.tm_mday)+'_'+str(T.tm_hour)+'_'+str(T.tm_min)+'_'+str(T.tm_sec)
    kwargs['clientName'] = clientName
    kwargs['timeOfCreation'] = Tstr
    kwargs['starttime'] = kwargs['starttime'].ctime()
    kwargs['endtime'] = kwargs['endtime'].ctime()
    kwargs['api'] = 'obspy'
    with open(readmePath, 'w') as readmeHandle:
        json.dump(kwargs, readmeHandle, indent=2)

def createFolderForSavingCat(containingFolderPath, clientName):

    #---- create the relevant name for the catalog
    # the naming convention should be [clientName]_[serial_number]
    #-- check for the serial number in the relevant folder
    folderNameRegex = clientName+'_[\d]*'
    Re = re.compile(folderNameRegex)
    relevantFolders = [p.name for p in Path(containingFolderPath).rglob('*') if bool(Re.match(p.name))]
    intsList = [int(r[len(clientName)+1:]) for r in relevantFolders]
    if len(intsList)==0:
        newIdx = 1
    else:
        newIdx = max(intsList) +1
    newFolderPath = Path(containingFolderPath) / (clientName+'_{}'.format(newIdx))
    return newFolderPath


def readCatAndMeta(containingFolder):

    #--- load meta file with creation details
    detailsPath = Path(containingFolder / 'catDetails.txt')
    with open(detailsPath,) as detHand:
        details = json.load(detHand)

    #--- load cat file
    cat_path = MyGeneral.files_of_certain_pattern(containingFolder, pattern='cat.*')[0]
    txtCat = Path(cat_path).suffix=='.txt'
    if txtCat:  # if it was created with saveAsTxt=True @ createAndSaveCat
        catDf = pd.read_csv(cat_path, sep=' ', header=0, dtype=txtCatColumnTypes(), na_values='None')
        catData = catDf.values
        columnNames = catDf.columns
        return (catData, columnNames), details
    else: # if saveAsTxt=False
        catRead = read_events(cat_path)
        return catRead, details

def txtCatColumnTypes():
    # Cat types for if it was created with saveAsTxt=True @ createAndSaveCat
    columnTypes = {
        'year': np.int, 
        'month': np.int, 
        'day': np.int, 
        'hour': np.int, 
        'minute': np.int, 
        'second': np.int, 
        'longitude': np.float64, 
        'longitude_err': np.float64, 
        'latitude': np.float64, 
        'latitude_err': np.float64, 
        'depth': np.float64, 
        'depth_err': np.float64, 
        'mag': np.float64, 
        'mag_error': np.float64, 
        'magnitude_type': str
    }
    return columnTypes

if __name__=="__main__":
    path = '/mnt/g/My Drive/Projects/MagMl/data/EQ_catalogs/NCEDC_1'
    files_list = MyGeneral.files_of_certain_pattern(path, pattern='cat.*')
    F = Path(files_list[0]).suffix=='.txt'
    F.suffix
    pass