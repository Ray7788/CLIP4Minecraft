import numpy as np

# TODO: RE-IMPLEMENT THIS FUNCTION
def compute_metrices(x):
    """ 
    Compute x@x metrices for a given array of scores
    """
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    return {'R1': float(np.sum(ind == 0)) * 100 / len(ind),
            'R5': float(np.sum(ind < 5)) * 100 / len(ind),
            'R10': float(np.sum(ind < 10)) * 100 / len(ind),
            'MedianR': np.median(ind) + 1,
            'MeanR': np.mean(ind) + 1,
            }
