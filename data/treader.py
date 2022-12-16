import numpy as np
from scipy.io import loadmat



def read_transients(filename):

    data = loadmat(filename)

    #print(data)

    transients = data['out_transients']
    #print(final_out.shape)

    t = transients[0][0][0].flatten()
    ppm = transients[0][0][1].flatten()
    fid1 = transients[0][0][2]
    fid2 = transients[0][0][3]

    return {
        "t":t,
        "ppm":ppm,
        "fid1":fid1,
        "fid2":fid2
    }

