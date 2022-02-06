import pyUtils
import numpy as np


if __name__=="__main__":

    pyUtils.MyGeneral.cachePickleDumpTo(**{'theta':2})
    pass
    R = np.random.rand(4,20)*120-4
    display_array = np.array([pyUtils.MySignal.normalizeToRange(r.data) for r in R])
    pass