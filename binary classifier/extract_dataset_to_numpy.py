import os
import numpy as np
import pandas as pd
import librosa

dataset_dir = "X:/CinC Challange 2016/"

X_dict = {}
for folder in os.listdir(dataset_dir):
    if folder.startswith("training-"):
        for f in os.listdir(dataset_dir + folder):
            if f.endswith(".wav"):
                X_k = librosa.load(dataset_dir + folder + "/" + f, sr=None)[0]
                X_k /= np.max(np.abs(X_k))
                X_dict[f.split(".")[0]] = X_k
                


Y_dict = {}
annotations_dir = dataset_dir + "annotations/updated/"
for folder in os.listdir(annotations_dir):
    if folder.startswith("training-"):
        for f in os.listdir(annotations_dir + folder):
            if f.endswith(".csv"):
                data = pd.read_csv(annotations_dir + folder + "/" + f, header=None).iloc[:,:2]
                for i,e in enumerate(data.iloc[:,0]):
                    Y_dict[e] = data.iloc[i,1]


## remove keys not both in X and Y dict
deleteX, deleteY = [], []
for k in X_dict.keys():
    if k not in Y_dict.keys():
        deleteX.append(k)
for k in Y_dict.keys():
    if k not in X_dict.keys():
        deleteY.append(k)
for k in deleteX:
    del X_dict[k]
for k in deleteY:
    del Y_dict[k]


print('X num recordings (pre augmentation):', len(X_dict))
print('num normal recordings:', len([y_i for y_i in Y_dict.values() if y_i==-1]))
# make all recordings length 5s/10000samples
X_dict_5 = {}
Y_dict_5 = {}
i = 0
for k in X_dict.keys():
  L = X_dict[k].shape[0]
  N = int(np.floor(L/10000))
  for win in range(N):
    X_dict_5[i] = X_dict[k][win*10000:(win+1)*10000]
    Y_dict_5[i] = Y_dict[k]
    i += 1



## make the X size:(num recordings, time steps, 1), Y size: (num recordings)

y = np.array(list(Y_dict_5[k] for k in Y_dict_5.keys()))
X = np.vstack(list(X_dict_5[k] for k in X_dict_5.keys()))

print('X shape:', X.shape)
print('y shape: ', y.shape)
np.save('X.npy', X)
np.save('y.npy', y)