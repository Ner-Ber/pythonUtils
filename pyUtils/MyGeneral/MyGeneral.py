#------------------------------------
# the functions below should not require any imports from my codes.
#------------------------------------

import sys
import copy
from copy import deepcopy
import os
import glob
import collections
import pickle
import numpy as np
from numpy.core.fromnumeric import ndim


def whos(var, doReturn=False):
    """
    like matlab whos
    """
    TYPE = type(var)
    SIZE = sys.getsizeof(var)
    try:
        SHAPE = var.shape
    except Exception:
        try:
            SHAPE = var.__len__()
        except Exception:
            SHAPE = None
    whosString = 'type {} \t  size {} \t shape {}'.format(TYPE, SIZE, SHAPE)
    if doReturn:
        return whosString
    print(whosString)

def whosDict(D):
    for n in D.keys():
        print('{}: \t\t {}'.format(n, whos(D[n], doReturn=True)))

def dictWhos(dict):
    """
    summerize dict''s coentent
    """
    for k,v in dict.items():
        whosString = whos(v, doReturn=True)
        print(k+': \t '+whosString)

def dict2string(D, sepMain='_', sepSec=''):
    STR = ''
    for (k,v) in D.items():
        STR += sepMain+k+sepSec+str(v)
    return STR[len(sepMain):]

def flattenList(List):
    typeIconsiderAsLists = [list, np.ndarray]

    flat_list = []
    for l in List:
        if (type(l) in typeIconsiderAsLists):
            flat_list = flat_list + flattenList(l)
        else:
            flat_list.append(l)
    return flat_list


def copyListStructure(List, key=None, replace=False, actOn=False):
    if key is None:
        key = lambda x: None
    if replace:
        newList = List
    else:
        newList = [None]*len(List)
    for l in range(len(List)):
        if type(List[l])==list:
            newList[l] = copyListStructure(List[l], key=key)
        else:
            try:
                if actOn:
                    key(List[l])
                else:
                    newList[l] = key(List[l])
            except:
                try:
                    if actOn:
                        for Ll in List[l].get_children(): key(Ll)
                except:
                    print('cant operate key({})'.format(str(List[l])))
                    if actOn:
                        pass
                    else:
                        newList[l] = List[l]
    if actOn:
        pass
    else:
        return newList


def flattenDict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flattenDict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def inverseDict(d):
    flipped = collections.defaultdict(dict)
    for key, val in d.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return flipped

def arange_like(array:np.array):
    """create a matrix composed of arange arrays.

    Args:
        array (np.array): N*M. N signals of length M

    Returns:
        np.array: N*M. each row (n) is a np.arange(M) vector.
    """
    assert array.ndim>2
    if array.ndim==1:
        return np.arange(array.size)

    dim0, dim1 = array.shape
    XX, _ = np.meshgrid(np.arange(dim1), np.ones(dim0))
    return XX




def files_of_certain_pattern(path, pattern='*'):
    fullPath = os.path.join(path, pattern)
    return glob.glob(fullPath)  # * means all if need specific format then *.csv

def newest_file_in_dir(path, spec='*'):
    list_of_files = files_of_certain_pattern(path, pattern=spec)
    return max(list_of_files, key=os.path.getctime)


# pickeCacheFile = 'C:/Users/NB/AppData/Local/Temp/python_pickle_cahce.pkl'
# pickeCacheFile = 'C:/Users/user/AppData/Local/Temp/python_pickle_cahce.pkl'
pickeCacheFile = '/tmp/python_pickle_cahce.pkl'

def cachePickleDumpTo(**Dict):
    outfile = open(pickeCacheFile,'wb')
    pickle.dump(Dict, outfile)
    outfile.close()

def cachePickleReadFrom():
    """
    use locals().update(new_load) after usage of function in order to update the variables in worksapce.
    or  locals().update(cachePickleReadFrom) if you aren't afraid of running over variables
    """
    infile = open(pickeCacheFile, 'rb')
    new_load = pickle.load(infile)
    infile.close()
    return new_load


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))