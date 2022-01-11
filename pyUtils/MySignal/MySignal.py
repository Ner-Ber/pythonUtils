import numpy as np
from numpy.lib import arraypad
from scipy import signal, optimize
import sys
from copy import deepcopy
# sys.path.append('G:\\My Drive\\pythonCode')
from  .. import MyGeneral



def whereLogicalRegion(booleanVec):
    regionsList = []
    newRegion = True
    for e,ii in enumerate(booleanVec):
        if ii & newRegion:
            startingRegion = e
            newRegion = False
        if ~newRegion & ~ii:
            endRegion = e
            newRegion = True
            regionsList.append([startingRegion, endRegion])
    if ii & newRegion:
        endRegion = e
        regionsList.append([startingRegion, endRegion])

    return regionsList


def createGaussianAprox1D(n,s=0.25):
    # createGaussianAprox creates a 1D binomial coefficient matrix (gaussian
    # approximation)
    # matrix's size will be (kernelSize,)
    x = np.linspace(-1,1,n)/s
    g = np.exp(-(x**2)/2)
    g /= g.sum()
    return g


def createGaussianAprox2D(n,s=0.25):
    # createGaussianAprox creates a 2D binomial coefficient matrix (gaussian
    # approximation)
    # matrix's size will be (kernelSize,kernelSize)

    g = createGaussianAprox1D(n,s)
    g = g[:,None]
    g_notNorm = signal.convolve2d(g, g.T)  # creating a kernel matrix
    g_notNorm /= g_notNorm.sum()
    return g_notNorm

    

def detrendNonZero(data:np.ndarray, epsilon:float=1e-6, deg=1):
    if ((len(data.shape)==1) | (1 in data.shape)):
        data = deepcopy(data.ravel()[None,:])
    else:
        data = deepcopy(data)
    nonZeroLogical = data>=epsilon
    #--- stricktly zero if smaller than epsilon
    data[~nonZeroLogical] = 0.
    #--- fit to the non zero 
    data_detrend = np.zeros_like(data)
    x = np.arange(data.shape[1])
    for e,d in enumerate(data):
        if ~np.any(nonZeroLogical[e]):
            continue
        p = np.polyfit(x[nonZeroLogical[e]],d[nonZeroLogical[e]], deg)
        data_detrend[e,nonZeroLogical[e]] = d[nonZeroLogical[e]] - np.polyval(p, x[nonZeroLogical[e]])
    
    return data_detrend

def convolveWnans(array:np.array, kernel:np.array, mode='full', padVal=0., flip_kernel=True):
    """calculate convolution of (multiple) 1D signals with missing data points in the form of nans.

    Args:
        array (np.array): MxN array. M is the number of signals, N is the signals' lengths. may Inclide nans.
        kernel (np.array): a 1D kernel to convolve with. Will be normalized each convolution step to account for nans.
        mode (str, optional): 'full'/'valid'/'same', same as meaning as in scipy.signal.convolve. Defaults to 'full'.
        padVal (float, optional): policy of edges. can be 'extend', 'cyclic', 'reflect' or a value. Defaults to 0.
        flip_kernel (bool, optional): Unfliped will result in correlation rather than convolution. Defaults to True.

    Returns:
        Mxn array. n is determined by the 'mode' parameter. see scipy.signal.convolve
    """

    k_len = kernel.size
    Npad = k_len - 1

    #-- set array to right size
    if len(array.shape)==1:
        arrayResh = array[:,None]
    else:
        arrayResh = deepcopy(array)
    Nsignals, signalLen = arrayResh.shape

    #-- pad the array and prepare:
    if padVal == 'extend':
        padLeft = np.repeat(arrayResh[:,:1], Npad, axis=1)
        padLeft = np.repeat(arrayResh[:,-1:], Npad, axis=1)
    elif padVal == 'reflect':
        padLeft = np.flip(arrayResh[:,:Npad], axis=1)
        padRight = np.flip(arrayResh[:,-Npad:], axis=1)
    elif padVal == 'cyclic':
        padRight = np.flip(arrayResh[:,:Npad], axis=1)
        padLeft = np.flip(arrayResh[:,-Npad:], axis=1)
    else:  # default or value
        padLeft = np.full((Nsignals, Npad), padVal)
        padRight = np.full((Nsignals, Npad), padVal)
    arrayPad = np.concatenate((padLeft, arrayResh, padRight), axis=1)

    #-- flip kernel (corelation vs convolution)
    if flip_kernel:
        kern_flip = np.flip(kernel)
    else:
        kern_flip = deepcopy(kernel)

    kerArray = np.repeat(kern_flip.ravel()[None,:], Nsignals, axis=0)


    ## iterae on all elements in signal to convolve and operate
    resultLen = signalLen+2*(int(k_len/2))
    result = np.full((Nsignals, resultLen), np.nan)
    for i_start in range(resultLen):
        i_end = i_start+k_len
        arrayPiece = arrayPad[:,i_start:i_end]  # crop the relevant pece for this convolution step
        nanMask = np.isnan(arrayPiece)    # create a mask by which certain values will be considered in conv step
        #-- create a kernel array for this step with 0's where nans are, and renormalize it
        ker_i = deepcopy(kerArray)
        ker_i[nanMask] = 0.
        sumPerRow = ker_i.sum(axis=1)[:,None]
        sumPerRow[sumPerRow==0] = np.nan    # force nan where there's no signals at all
        ker_i /= sumPerRow
        result[:,i_start] = ((ker_i*arrayPiece).sum(axis=1))
    
    if mode=='full':
        return result
    elif mode=='valid':
        return result[:,k_len-1:(-k_len+1)]
    elif mode=='same':
        return result[:,int(k_len/2):(-int(k_len/2))]

def normalizeToRange(signal_array:np.ndarray, range=[-1,1]):
    if signal_array.ndim==1:
        one_dim = True
        signal_array = signal_array[None,:]+0
    else: 
        one_dim = False
        signal_array = signal_array+0

    signal_array -= signal_array.min(axis=1)[:,None]
    signal_array /= signal_array.max(axis=1)[:,None]
    signal_array *= np.ptp(range)
    signal_array += np.min(range)
    if one_dim:
        return signal_array.ravel()
    else:
        return signal_array

    
    pass



if __name__=="__main__":
    # locals().update(MyGeneral.cachePickleReadFrom())
    R = np.random.rand(4,20)*120-4
    display_array = np.array([normalizeToRange(r.data) for r in R])
    pass
