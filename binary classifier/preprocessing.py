import numpy as np
from emd import sift 

def minkowski(x,y):
  return np.linalg.norm(x-y)

# Returns the dataset denoised with EMD and distance filtered 
# with the euclidean (minkowski p=2) distance.
def emd_minkowski(x, k):
  out = []
  for i in range(x.shape[0]):
    imfs = sift(x[i,:]).T
    dists = [minkowski(x[i,:], imf) for imf in imfs]
    idx = np.argpartition(dists,k+1)[:k+1] # get the k smallest dist indicies
    out.append(np.sum([imfs[int(idx[i])] for i in range(idx.shape[0])], axis=0))
  return np.vstack(out)