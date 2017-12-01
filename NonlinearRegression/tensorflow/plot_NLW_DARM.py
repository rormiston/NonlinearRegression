import numpy as np
import pylab as plt

darm_STFT = np.load("STFT_darm.npy").T
NLW_STFT = np.load("NLW_estimate.npy")

# compute transfer function
csd_STFT = np.multiply(darm_STFT.conjugate(), NLW_STFT)

# take time averages
csd_STFT = np.mean(csd_STFT, 0)
NLW = np.mean(np.abs(NLW_STFT)**2, 0)
DARM = np.mean(np.abs(darm_STFT)**2, 0)

C = np.abs(csd_STFT)**2 / NLW / DARM

plt.semilogx(C, 'o', label='coherence')
plt.legend()
