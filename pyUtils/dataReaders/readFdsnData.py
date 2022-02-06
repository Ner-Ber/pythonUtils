#!"C:\Users\user\.conda\envs\generalEnv_2\python.exe"

from numpy.core.fromnumeric import trace
import pandas as pd
from collections import defaultdict
import numpy as np
import itertools
from tqdm.notebook import tqdm
from scipy import signal, stats, optimize
import matplotlib.pyplot as plt
import obspy
from obspy.clients.fdsn import Client
from obspy.core import read, UTCDateTime, Stream
from obspy.signal.util import util_geo_km
from obspy.signal.trigger import classic_sta_lta
import sys
sys.path.append('G:\\My Drive\\pythonCode')
import MySignal





def readFedcatalogInfoFile(file_path):
    #--- read first line for headers:
    with open(file_path) as f:
        headerLine = f.readline()
        headerLine = headerLine.replace('#','')
        headerLine = headerLine.replace(' ','')
        headerLine = headerLine.replace('\n','')
        headers = headerLine.split('|')
    return pd.read_csv(file_path, sep='|', comment='#', names=headers)

def createBulkStringsFromFdsnDataFile(file_path):
    stations_data_df = readFedcatalogInfoFile(file_path)
    #--- required in bulk string: network, station, location, channel, starttime and endtime
    #--- according to https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms_bulk.html
    stations_data_df.columns = stations_data_df.columns.str.lower()
    relevant_cols = ['network','station','location','channel','starttime','endtime']
    return stations_data_df.to_string(header=False, index=False, columns=relevant_cols, na_rep='*')

# %% TODO: make these function work from "seisHelperFunction.py" not from here where they don't belong
def calcCC(channel_names, station_names, id_list, stations_DF, data_array):
    CC_result = defaultdict(lambda: [])
    for ch in channel_names:
        for station_pair in tqdm(itertools.combinations(station_names, 2)):
            #--- calc distance between stations
            pair_logical = (stations_DF['station'].values==station_pair[0]) | (stations_DF['station'].values==station_pair[1])
            pair_dist = np.sqrt((np.diff(stations_DF.loc[pair_logical,['x_m', 'y_m']].values, axis=0)**2).sum())
            #-- what specific channels will I be working with
            pair_ids = [f'{stat}..{ch}' for stat in station_pair]
            #--- get the indexes of these specific channels
            pair_idxs = [e for e,id in enumerate(id_list) if (id in pair_ids)]
            if (len(pair_idxs)<2): continue
            #--- preform CC;
            in_0 = data_array[pair_idxs[0]].ravel()
            in_1 = data_array[pair_idxs[1]].ravel()

            CC = signal.correlate(in_0, in_1/in_1.size).ravel()
            CC_lags = signal.correlation_lags(in_0.size, in_1.size)
            max_coor_val = CC.max()
            max_coor_lag = CC_lags[CC.argmax()]

            CC_result[ch].append(
                {
                    'd':pair_dist,
                    'pair':station_pair,
                    'max_coor_val':max_coor_val,
                    'max_coor_lag':max_coor_lag,
                }
            )
        CC_result[ch] = pd.DataFrame(CC_result[ch])
    return CC_result


def returnCleanStreamFromDowloadedFile(waveforms_pathlist):
    #-- read waveforms files
    st = obspy.Stream()
    print('reading waveforms from file...')
    for p in tqdm(waveforms_pathlist):
        st += obspy.read(p)

    #-- discard all LOG channels, they interrupt and I don't know what to do with them
    ross_streams = obspy.Stream()
    log_counter = 0
    for s in st:
        s_id = s.get_id()
        if not s_id.lower().endswith('log'):
            ross_streams += s
        else:
            log_counter += 1
    print(f'logs discarded = {log_counter}')

    #--- merge the rest

    print('merging common waveforms in stream')
    ross_streams.merge();

    #-- read inventory to clean events:
    #--- waveforms catalog metadata (dowloaded from fdsn website: http://service.iris.edu/irisws/fedcatalog/1/query?net=2C&starttime=2010-01-01&endtime=2011-12-31&format=text&includeoverlaps=true&nodata=404
    # link can be found here: https://www.fdsn.org/networks/detail/2C_2010/)
    C2_network_data = 'G:\\My Drive\\pythonCode\\icequakes\\seismic_analysis\\dataAndInfo\\irisws-fedcatalog_2021-12-26T09 19 51Z.text'
    stations_bulk = createBulkStringsFromFdsnDataFile(C2_network_data) # a string containing all relevant data
    #--- turn string into lists so all metadata can be dowloaded (an obspy thing, see warning "UserWarning: Parameters odict_keys(['level', 'includerestriced', 'includeavailability']) are ignored when request data is provided as a string or file!")
    last2args2UTC = lambda str_list: str_list[:-2] + [UTCDateTime(str_list[-2]),UTCDateTime(str_list[-1])]
    bulk_4_stations = [last2args2UTC(bulk.split(' ')) for bulk in stations_bulk.split('\n')]

    #--- downloading metadata
    print('download stations metadata')
    dataCenter="IRIS"
    client = Client(dataCenter)
    inv = client.get_stations_bulk(bulk_4_stations, level='response')

    print('samplt of inventory:')
    R = inv[0].get_response('2C.BB01..HHZ', UTCDateTime('2010-12-03T01:01:12'))
    print(R)

    print('removing sensitivity:')
    ross_streams.remove_sensitivity(inv)
    print(f'number of steams: {len(ross_streams)}')

    return ross_streams, inv

def createStationDf(waveform_streams, stations_inv, ref_lon_lat=None):
    coord_data = []
    for rsi in waveform_streams:
        id_seed = rsi.get_id()
        corrs_dict = stations_inv.get_coordinates(id_seed)
        corrs_dict['station'] = '.'.join(id_seed.split('.')[:2])
        coord_data.append(corrs_dict)

    if (ref_lon_lat is None):
        if False:
            N = len(coord_data)
            mean_lon = 0
            mean_lat = 0
            for i in coord_data: mean_lon+=i['longitude'] ; mean_lat+=i['latitude']
            mean_lon /= N
            mean_lat /= N
            ref_lat = mean_lat
            ref_lon = mean_lon
        else:
            mins = np.array([[i['longitude'], i['latitude']] for i in coord_data]).min(axis=0)
            ref_lon, ref_lat = mins[0], mins[1]
    else:
        ref_lon, ref_lat = ref_lon_lat

    for i in coord_data: i['x_m'], i['y_m'] = (1e3*km for km in util_geo_km(ref_lon,ref_lat,i['longitude'],i['latitude']))
    return pd.DataFrame(coord_data).drop_duplicates(ignore_index=True)

def returnNucleation(high_or_low='H'):
    if high_or_low.lower().startswith('H'):
        lat = -84.4
        lon = -157
    elif high_or_low.lower().startswith('L'):
        lat = -84.55
        lon = -163
    else:
        lat = -83.6
        lon = -159
    return lon, lat


def plotMoveout(ax, traces, time_vecs, distances, scale_factor=1, colors=None):
    handles = []
    for ii in range(len(traces)):
        T = time_vecs[ii].ravel()
        V = (MySignal.normalizeToRange(traces[ii])+distances[ii]*(scale_factor)).ravel()
        cl = None if (colors is None) else colors[ii]
        p = ax.plot(T,V,color=cl);
        handles.append(p);
    #-- set the y ticks
    minD = np.min(distances)
    maxD = np.max(distances)
    dtick = np.round((np.round(maxD) - np.round(minD))/10, -3)
    dist_ticks = np.arange(np.round(minD,-3), np.round(maxD,-3), dtick).astype(int)
    y_ticks = dist_ticks*scale_factor
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(dist_ticks)
    return handles



def getQ_nastyWay(stream, freqs, stations_df, omega):

    get_d = lambda station_name: np.sqrt((stations_df.loc[(stations_df['station']==station_name),['x_m','y_m']].values**2).sum())
    get_d_from_trace = lambda tr: get_d('.'.join(tr.get_id().split('.')[:2])  )
    dist_list_from_stream = lambda st: [get_d_from_trace(tr) for tr in st]
    get_xy = lambda station_name: tuple(stations_df.loc[(stations_df['station']==station_name),['x_m','y_m']].values.ravel())
    get_xy_from_trace = lambda tr: get_xy('.'.join(tr.get_id().split('.')[:2])  )
    xy_array_from_stream = lambda st: np.array([get_xy_from_trace(tr) for tr in st])
    data_array_from_stream = lambda st: np.array([tr.data for tr in st])
    times_array_from_stream = lambda st: np.array([tr.times() for tr in st])
    station_names_from_stream = lambda st: ['.'.join(tr.get_id().split('.')[:2]) for tr in st]


    #--- filter and calc energy:
    filter_size = 5001
    conv_kernel = MySignal.createGaussianAprox1D(filter_size,0.5)
    energy_array_from_stream = lambda st: np.array([np.convolve(v.data**2, conv_kernel) for v in st])
    E_array = energy_array_from_stream(stream)
    N = len(stream)
    
    E_array_crop = E_array+0
    E_array_crop = MySignal.normalizeToRange(E_array_crop[:,12500:100000],[0,1])
    times_cropped = times_array_from_stream(stream)[:,12500:100000]

    # crop: 
    # 15 before
    # 52 after
    peak_idxs = np.argmax(E_array_crop, axis=1)
    crop_limits = np.array([peak_idxs-15000, peak_idxs+52000])
    diffusion_cropped = []
    diffusion_times_cropped = []
    for i in range(peak_idxs.size):
        diffusion_cropped.append(E_array_crop[i][crop_limits[0][i]:crop_limits[1][i]])
        diffusion_times_cropped.append(times_cropped[i][crop_limits[0][i]:crop_limits[1][i]])


    Q_result_vec = []
    for i in range(N):
        work_data = diffusion_cropped[i]
        if (not work_data.size): Q_result_vec.append(np.nan); continue
        one_idx = np.argmin(np.abs(diffusion_cropped[i]-1))
        roll_var = pd.Series(work_data).rolling(10000).std().values
        work_data_stretch = work_data/((1-2*roll_var[one_idx]))
        ones_like = np.ones_like(work_data_stretch)-roll_var[one_idx]
        t_peak = np.mean(np.argwhere(np.diff(np.sign(work_data_stretch - ones_like))))

        regress2 = np.log(work_data_stretch[int(t_peak+0.5):])
        t_4regress = diffusion_times_cropped[i][int(t_peak+0.5):]
        res = stats.linregress(t_4regress, regress2)
        Q_result_vec.append(-omega/res.slope)

    return Q_result_vec




if __name__=="__main__":
    import sys
    sys.path.append('G:\\My Drive\\pythonCode')
    import MyGeneral

    locals().update(MyGeneral.cachePickleReadFrom())
    CC = calcCC(channel_names, station_names, id_list, stations, array_trimmed)
    

    # file_path = 'G:\\My Drive\\pythonCode\\icequakes\\seismic_analysis\\dataAndInfo\\irisws-fedcatalog_2021-12-26T09 19 51Z.text'
    # createBulkStringsFromFdsnDataFile(file_path)
    pass
