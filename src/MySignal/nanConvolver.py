import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal, stats

class nanConvolver:
    def __init__(self, Y, *X):
        """

        :param Y: data values
        :param X: data control parameter (optional)
        """
        self.load_data(Y,X)


    def load_data(self, Y, X):
        #     """
        #     :param Y: data values
        #     :param X: data control parameter (optional)
        #     """
        if isinstance(Y, pd.Series) | isinstance(Y, pd.DataFrame):
            self.Y = Y.values
        else:
            self.Y = Y

        if len(X) == 0:
            self.X = np.arange(1, len(Y) + 1, dtype=float)
        else:
            if isinstance(X[0], pd.Series) | isinstance(X[0], pd.DataFrame):
                self.X = X[0].values
            else:
                self.X = X[0]
        pass

    def createKernel(self, kerStyle, windowLength, *widthParameter):
        """

        :param kerStyle:            name of window to convolve with: 'gaussian', 'uniform', 'lognorm', 'chi2', 'power', 'uniform_past'
        :param windowLength:    number of pixels of window (kernel)
        :param widthParameter:  additional parameter defining the window, typically width, e.g. std in 'gauss'
        :return:                kernel NOT NORMALIZED
        :
        : the 'power' distribution is defined by y=(L+x)^a, -L<=x<=0, a>0. width of this distribution is defined by the
        : FWHM of it = L(1-2^(-1/a))
        """
        ## define width parameter if necessary
        if len(widthParameter) == 0:
            widthParameter = np.floor(windowLength / 4)
        else:
            widthParameter = widthParameter[0]
        windowLength = float(windowLength)
        widthParameter = float(widthParameter)

        ## define kernel
        if (kerStyle.lower() == 'gaussian') | (kerStyle.lower() == 'gauss'):
            kernel = signal.gaussian(windowLength, widthParameter)
        elif (kerStyle.lower() == 'uniform') | (kerStyle.lower() == 'flat'):
            kernel = np.ones(int(windowLength), dtype=float)
        elif (kerStyle.lower() == 'uniform_past') | (kerStyle.lower() == 'flat_past'):
            half_kern = np.ones(int(windowLength), dtype=float)
            func_and_zeros = np.concatenate((half_kern, np.full(int(windowLength) - 1, 0.)))
            kernel = np.flip(func_and_zeros, axis=0)
        elif (kerStyle.lower() == 'lognorm'):
            kernel = np.flip(stats.lognorm.pdf(np.arange(windowLength), s=widthParameter), axis=0)
        elif (kerStyle.lower() == 'chi2'):
            kernel = np.flip(stats.chi2.pdf(range(int(windowLength)), s=widthParameter / 2), axis=0)
        elif (kerStyle.lower() == 'power'):
            alpha = (np.log2((windowLength + widthParameter) / windowLength)) ** (-1)
            power = lambda x: x ** alpha
            func_and_zeros = np.concatenate((power(np.arange(windowLength)), np.full(int(windowLength) - 1, 0.)))
            kernel = np.flip(func_and_zeros, axis=0)
        elif (kerStyle.lower() == 'exp'):
            expFunc = lambda x: np.exp(x / widthParameter)
            func_and_zeros = np.concatenate((expFunc(np.arange(windowLength)), np.full(int(windowLength) - 1, 0.)))
            kernel = np.flip(func_and_zeros, axis=0)
        else:
            raise ('unrecognized kernel style')

        return kernel

    def loadKernel(self, kernel):
        """
        loadKernel(self, kernel):
        :param kernel: a numpy 1D array with the desired distribution. not normalized. May also be a tuple of string indicating which kernel to create. in This case 'loadKernel' calls 'createKernel'
        :return: saves the kernel to self
        """

        ## create kernel if not given, otherwise load it
        if (type(kernel) == tuple) & (len(kernel) == 2):
            kerStyle, windowLength = kernel
            self.kernel = self.createKernel(kerStyle, windowLength)
            self.kernelDetails = (kerStyle, windowLength, np.floor(windowLength / 4))
        elif (type(kernel) == tuple) & (len(kernel) == 3):
            kerStyle, windowLength, widthParameter = kernel
            self.kernel = self.createKernel(kerStyle, windowLength, widthParameter)
            self.kernelDetails = (kerStyle, windowLength, widthParameter)
        elif type(kernel) == np.ndarray:
            self.kernel = kernel
            self.kernelDetails = ('costume')

    def nanConvolve(self, flag='same', padVal=np.nan, normPerFrame=False, flipKernel=True):
        
        k_len = self.kernel.size
        Npad = k_len - 1
        L = self.X.size

        ## pad data and prepare:
        if padVal == 'extend':
            padLeft = np.full((Npad), self.Y[0])
            padRight = np.full((Npad), self.Y[-1])
        elif padVal == 'reflect':
            padLeft = np.flip(self.Y[0:Npad].reshape((1, -1)), axis=1)
            padRight = np.flip(self.Y[-Npad:].reshape((1, -1)), axis=1)
        elif padVal == 'cyclic':
            padLeft = np.flip(self.Y[-Npad:].reshape((1, -1)), axis=1)
            padRight = np.flip(self.Y[0:Npad].reshape((1, -1)), axis=1)
        else:  # default or value
            padLeft = np.full((Npad), padVal)
            padRight = np.full((Npad), padVal)


        Ypadded = np.concatenate((padLeft.reshape((1, -1)), self.Y.reshape((1, -1)), padRight.reshape((1, -1))), axis=1)
        ## create new X axis:
        X_interped = self.create_new_X_axis(Ypadded.size, Npad)


        NpadLeft = int(np.floor(float(k_len) / 2) + (float(k_len) % 2 - 1))
        NpadRight = int(np.floor(float(k_len) / 2))

        ## iterae on all elements in signal to convolve and operate
        if flipKernel:
            kern_final = np.asarray(np.fliplr(self.kernel.reshape((1, -1))),
                                    dtype=float)  # flip kernel as in conv definiton and make sure kernel is float
        else:
            kern_final = np.asarray(self.kernel.reshape((1, -1)),
                                    dtype=float)  # flip kernel as in conv definiton and make sure kernel is float

        Yresult = np.full(Ypadded.size, np.nan)
        numDataPoints = np.full(Ypadded.size, np.nan)
        distBeforeMiddle = np.full(Ypadded.size, np.inf)

        allTrue = np.full((1, NpadLeft + NpadRight + 1), True)  # size of padds, only Trues. operetional resons for filtering later on
        for i in range(NpadLeft, Ypadded.size - NpadRight):
            Y_piece = Ypadded[:,
                      range(i - NpadLeft, i + NpadRight + 1)]  # take the piece of which this convolution step works on
            thisLogic = allTrue & (np.isfinite(Y_piece) + (
                not normPerFrame))  # dfine (in a logical vector) weather each element in the piece is included in normalization
            kNorm = kern_final / np.sum(kern_final[thisLogic])  # normalize accordint to descision in previous step
            multip = kNorm * Y_piece  # multiply the normlized kernel and the piece
            Yresult[i] = np.nansum(multip) * (np.nan if np.isnan(
                multip).all() else 1.)  # sum the result ingnoring nans. in the entire result is nans, return nan
            numDataPoints[i] = np.sum(np.isfinite(Y_piece) & (kern_final != 0))
            finiteLocs = np.ravel(np.where(np.isfinite(Y_piece).ravel())) - NpadLeft
            if np.sum(finiteLocs <= 0) > 0:
                distBeforeMiddle[i] = np.max(finiteLocs[finiteLocs <= 0])
            else:
                distBeforeMiddle[i] = np.inf

        return self.trim_by_flag(flag, Yresult, X_interped, numDataPoints, distBeforeMiddle, k_len, NpadLeft, NpadRight)

    def create_new_X_axis(self, newLen, Npad):
        x_diff = np.diff(self.X)
        ref_idx = np.where(~np.isnan(self.X))[0][0]
        ref_idx_newLoc = ref_idx+Npad
        X_new = np.arange(newLen, dtype=float) * np.nanmin(x_diff)
        X_new += (self.X[ref_idx] - X_new[ref_idx_newLoc])
        return X_new

    def trim_by_flag(self, flag, Yresult, X_interped, numDataPoints, distBeforeMiddle, k_len, NpadLeft, NpadRight):

        if flag == 'full':
            start, end = None,  None
        elif flag == 'valid':
            start = (k_len + NpadLeft - 1)
            end = -(k_len + NpadRight - 1)
        else:
            start =(k_len - 1)
            end = -(k_len - 1)
        start = start if start>=0 else 0    # I dont know why this would happen, but just to be sure
        end = end if end<=-1 else None      # this can happen when you dont want to trim anything from the right so you get end=0

        self.Yconvolved = Yresult[start:end]
        self.numDataPoints = numDataPoints[start:end]
        self.distBeforeMiddle = distBeforeMiddle[start:end]
        self.X_interped = X_interped[start:end]

        return self.Yconvolved