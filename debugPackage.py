import src
import numpy as np


if __name__=="__main__":

    R = np.random.rand(4,20)*120-4
    display_array = np.array([src.MySignal.normalizeToRange(r.data) for r in R])
    pass