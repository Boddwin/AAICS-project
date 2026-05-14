import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc_lib


def calculate_delta(array):
    rows, _ = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            first  = max(0, i - j)
            second = min(rows - 1, i + j)
            index.append((second, first))
            j += 1
        deltas[i] = (
            array[index[0][0]] - array[index[0][1]]
            + 2 * (array[index[1][0]] - array[index[1][1]])
        ) / 10
    return deltas


def extract_features(audio, rate):
    """Return a (frames, 40) array of MFCC + delta features."""
    mfcc_feat = mfcc_lib.mfcc(audio, rate, 0.025, 0.01, 20,
                               appendEnergy=True, nfft=1201)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    return np.hstack((mfcc_feat, delta))
