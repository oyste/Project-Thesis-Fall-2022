import matplotlib.pyplot as plt
import librosa
import numpy as np
data, sr = librosa.load('X:/BINARY CLASSIFIER/figure_making_code/a0011.wav', sr=2000)
plt.plot(np.linspace(1,4000,4000)/sr,data[5000:9000]/np.max(abs(data[5000:9000])))
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$Amplitude$', fontsize=18)
plt.show()