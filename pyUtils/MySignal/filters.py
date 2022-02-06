import numpy as np


def movingLinearFit(signal, win_len=101, padVal='reflect', mode='same'):
    sig_pad = signalPadder(signal, win_len, padVal=padVal, mode=mode)
    sig_pad = (sig_pad+0).ravel()
    sig_range = np.arange(sig_pad.size)
    num_iteration = sig_pad.size - (win_len-1)
    running_lin_fit = []
    for ii in range(num_iteration):
        section = sig_pad[ii:(ii+win_len)][:,None]
        section_range = sig_range[ii:(ii+win_len)][:,None]
        W = np.linalg.pinv(np.hstack((np.ones_like(section_range), section_range))) @ section
        running_lin_fit.append(tuple(W.ravel()))
    return running_lin_fit


def signalPadder(signal:np.ndarray, k_len, padVal='reflect', mode='full'):
    """pad a signal as preparation for processing with a kernel (convolution, running std, etc..)

    Args:
        signal (np.ndarray): a 1D signal, can have extra dims of length 1
        k_len ([type]): length of the kernel to be used
        padVal (str/numeric, optional): method of padding, can be the value of which to pad with, 'extend', 'reflect' ot 'cyclic'.
                                        Defaults to 'reflect'.

    Returns:
        np.ndarray: a 1D signal.
    """

    # TODO: add 'mode' option in aeguments to pad for 'same', 'extend' or 'valid (no padding)

    assert np.sum(np.array(signal.shape)!=1)==1
    if mode=='valid':
        return

    orig_shape = signal.shape
    signal = (signal+0).ravel()
    if mode=='full':
        Npad = k_len - 1
    elif mode=='same':
        Npad = int(k_len/2)
    else:
        assert False, "argument 'mode' invalid"
    L = signal.size

    ## pad data and prepare:
    if padVal == 'extend':
        padLeft = np.full((Npad), signal[0])
        padRight = np.full((Npad), signal[-1])
    elif padVal == 'reflect':
        padLeft = np.flip(signal[1:(Npad+1)])
        padRight = np.flip(signal[-(Npad+1):-1])
    elif padVal == 'cyclic':
        # padLeft = np.flip(signal[-Npad:].reshape((1, -1)), axis=1)
        # padRight = np.flip(signal[0:Npad].reshape((1, -1)), axis=1)
        padLeft = signal[-Npad:]
        padRight = signal[0:Npad]
    else:  # default or value
        padLeft = np.full((Npad), padVal)
        padRight = np.full((Npad), padVal)


    signal_padded = np.concatenate((padLeft, signal, padRight))
    return signal_padded.reshape([-1 if o!=1 else 1 for o in orig_shape])
 


if __name__=='__main__':


    import sys
    sys.path.append("G:\\My Drive\\pythonCode")
    import MyGeneral
    import matplotlib.pyplot as plt
    locals().update(MyGeneral.cachePickleReadFrom())

    V = data['n'].T
    x = np.arange(V[0].size)
    for v in V:
        run_lin_fit = movingLinearFit(v, 3001, mode='same')
        w0,w1 = zip(*run_lin_fit)
        hist, bin_edges = np.histogram(w1,100)
        bin_centers = np.convolve(bin_edges, [0.5,0.5], mode='valid')
        selected_slope = bin_centers[np.argmax(hist)]

        first_idx = np.where(np.isfinite(v))[0][0]
        v_rmv = v-v[first_idx]
        v_rmv -= selected_slope*(x-first_idx)

        
        pass


    S = np.round(np.random.rand(1, 30, 1)*100)
    run_lin_fit = movingLinearFit(S, 11)
    pass