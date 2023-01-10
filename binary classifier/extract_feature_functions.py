from kymatio.numpy import Scattering1D
import librosa
import numpy as np

def get_wsn_feats(x):
    T = 2**10
    J = 10
    Q = (8,1)
    shape = x.shape[-1]
    log_eps = 1e-6
  
    scattering = Scattering1D(shape=shape,J=J, T=T, Q=Q)
    sx_i = []
    for i in range(x.shape[0]):
        sx_i.append(scattering(x[i,:]))
        print(i)
        print(sx_i[i].shape)
    sx = np.stack(sx_i,axis=0)
    sx = sx[:,1:,:]
    sx = np.log(abs(sx)+log_eps)
    sx = np.mean(sx, axis=-1)
    return sx

def get_mfcc_feats(x):
    x_mfcc = []
    for i in range(x.shape[0]):
        mf = librosa.feature.mfcc(y=x[i,:], sr=2000, n_mfcc=15)
        d_mf = librosa.feature.delta(mf)
        dd_mf = librosa.feature.delta(d_mf)
        feats = np.concatenate((mf, d_mf, dd_mf), axis=0)
        x_mfcc.append(feats)
    x_mfcc = np.stack(x_mfcc, axis=0)
    flat_x_mfcc = []
    for i in range(x_mfcc.shape[0]):
      flat_x_mfcc.append(np.average(x_mfcc[i,:,:], axis=1))
    flat_x_mfcc = np.vstack(flat_x_mfcc)
    return flat_x_mfcc

def r(q_i, S_c, gamma):
  if q_i >=(S_c + gamma):
    return 1
  elif q_i <= (S_c - gamma):
    return -1
  return 0

def r_u(q_i, S_c, gamma):
  return int(r(q_i, S_c, gamma) == 1)

def r_l(q_i, S_c, gamma):
  return int(r(q_i, S_c, gamma) == -1)

def leq(a,b):
  return int(a<=b)

def kdelta(a,b):
  return int(a==b)

# resulting in L**2 size flattened feature vector 
def get_altp_feats(x, L=5):
  x_altp = []
  N = int(np.floor(x.shape[1]/L))
  for i in range(x.shape[0]):
    D_u, D_l = [], []
    #print('x_shape', x.shape[1])
    print(i)
    for j in range(N): # sliding window
        frame = x[i,j*L:(j+1)*L]
        gamma = np.mean(frame)/np.std(frame)
        mid_idx = int(np.floor((len(frame)-1)/2))
        #print(frame, mid_idx)
        S_c = frame[mid_idx]
        r_u_lst, r_l_lst = [], []
        for idx, q in enumerate(frame):
            if idx != mid_idx:
                r_u_i = r_u(q, S_c, gamma)
                r_l_i = r_l(q, S_c, gamma)
                r_u_lst.append(r_u_i)
                r_l_lst.append(r_l_i) 
        D_u.append(np.sum(r_u_lst*2**np.arange(L-1), axis=0)) # k=0 to L-2
        D_l.append(np.sum(r_l_lst*2**np.arange(L-1), axis=0))
    C_l_out, C_u_out = [], []
    for j in range(2**(L-1)):
      C_u_out.append(np.sum([kdelta(d_u,j) for d_u in D_u]))
      C_l_out.append(np.sum([kdelta(d_l,j) for d_l in D_l]))
    x_altp.append(np.hstack([np.hstack(C_l_out), np.hstack(C_u_out)]))
  return np.vstack(x_altp)

  