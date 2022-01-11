import numpy as np
from scipy import signal
from pathlib import Path
import sys
# sys.path.append(Path('G:\My Drive\pythonCode\MySignal'))
from .MySignal import *

class signalGaussianPyramid:
    def __init__(self, signal=None, signalTime=None, numPyrLevels=4, initGaussLength=51, gaussWidth=0.4, generatePyr=True) -> None:
        #--- set pyr params
        self.setGaussLength(initGaussLength)
        self.setGaussWidth(gaussWidth)
        self._numPyrLevels(numPyrLevels)
        self.setInputSignal(signal, signalTime)
        if generatePyr:
            self.generatePyrFromSignal()

    def setInputSignal(self, signal, signalTime=None):
        """store the 1D signals you'd like to generate a pyramid for.
        This will be saved directly in the 

        Args:
            signal (ndarray): A N*M matrix. N signals of length M
        """
        self.originalSig = signal
        if signalTime is None:
            signalTime = np.arange(self.originalSig.shape[0])
        self.signalTime = signalTime.ravel()

    def setGaussLength(self, gaussInitialLength=51):
        #-- force kernel to be odd length
        self._gaussInitialLength = int(gaussInitialLength + np.abs(np.mod(gaussInitialLength,2)-1))

    def setGaussWidth(self, gaussWidth=0.4):
        self._gaussWidth = gaussWidth

    def generatePyrFromSignal(self):
        self.signalPyr = {0: self.originalSig}
        self.timePyr = {0: self.signalTime}
        for i in range(1,self._nLevels):
            #-- create kernel for smoothing
            newLength = self._gaussInitialLength/(2**(i-1))
            newLength += 1- np.mod(newLength,2)
            kerLength = int(np.round(newLength))
            currentKer = createGaussianAprox1D(kerLength,self._gaussWidth)[None,:]
            #-- padd previous level to prepare for smoothing
            paddLength = int(np.floor(kerLength/2))
            paddedLevel = np.pad(self.signalPyr[i-1], ((0,0),(paddLength,paddLength)))
            #-- smooth previous level
            # smoothedLevel_im1 = signal.convolve(paddedLevel, currentKer, 'valid')
            smoothedLevel_im1 = convolveWnans(paddedLevel, currentKer, 'valid')
            #-- subsample level
            self.signalPyr[i] = smoothedLevel_im1[:,::2]
            self.timePyr[i] = self.timePyr[i-1][::2]

    def sigDerive(self, derivedSigName='vel', scaleName='time4vel', what2Derivate='signalPyr', what2Derivate_scale='timePyr'):
        """finite difference derivate a signal by it's relevant scale

        Args:
            derivedSigName (str, optional): name of the derivative. Defaults to 'vel'.
            scaleName (str, optional): name of the derivatives scale. Defaults to 'time4vel'.
            what2Derivate (str, optional): what pyramid to use as basis for derivation. Defaults to 'signalPyr'.
            what2Derivate_scale (str, optional): what scale to use as basis. Defaults to 'timePyr'.
        """
        scaleAtt = {}
        deriveSigAtt = {}

        sig2derivate = self.__getattribute__(what2Derivate)
        scale2derivate = self.__getattribute__(what2Derivate_scale)

        avgKernel = np.array([0.5, 0.5])
        for k,v in sig2derivate.items():
            scaleAtt[k] = np.convolve(scale2derivate[k], avgKernel,  mode='valid')
            halfDt = np.mean(np.diff(scale2derivate[k]))/2
            deriveSigAtt[k] = np.diff(v, axis=1)/halfDt
        
        self.__setattr__(derivedSigName, deriveSigAtt)
        self.__setattr__(scaleName, scaleAtt)

    def _numPyrLevels(self, n=4):
        self._nLevels = n

if __name__=='__main__':
    import sys
    from pathlib import Path
    # sys.path.append('G://My Drive//pythonCode')
    from .. import MyGeneral
    locals().update(MyGeneral.cachePickleReadFrom())

    numPyrLevels=3
    initGaussLength=51
    gaussWidth=0.4
    generatePyr=True


    e_pyrInst = signalGaussianPyramid(signal=data['e'].T, signalTime=data['t'],\
                numPyrLevels=numPyrLevels, initGaussLength=initGaussLength, gaussWidth=gaussWidth, generatePyr=generatePyr)
    e_pyrInst.signalPyr[1][5]
    pass
