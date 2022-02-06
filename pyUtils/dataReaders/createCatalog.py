#!C:\Users\NB\pyenvs\generalEnv_2\Scripts\python.exe python3

import quakeUtils
import obspy

folderPath = 'H:\My Drive\EQ_catalogs'
Ndays = 13*365
# Ndays = 5*365
startTime = obspy.UTCDateTime(2008,1,1)
catalogOption = {
    'starttime'     : startTime,
    'endtime'       : startTime + 60*60*24*Ndays,
    'latitude'      : 35.07,
    'longitude'     : -119.39,
    'maxradius'     : 4,
    'mindepth'      : 0,
    'maxdepth'      : 6000,
    'minmagnitude'  : 0,
    'maxmagnitude'  : 20,
    'magnitudetype' : None
}

clientName='NCEDC'
quakeUtils.createAndSaveCat(folderPath, clientName, **catalogOption)