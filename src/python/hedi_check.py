import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy.io as io
import numpy as np
from lasp.timefreq import gaussian_stft
from lasp.sound import plot_spectrogram
import pdb

srate = 41100
f = io.loadmat('final.fit.mbird3call3.mat')
tfsong = f['fsong'][0][0]

fsong = {
           'w': tfsong[0],
     'wavname': tfsong[1],
          'nw': tfsong[2],
      'r_moto': tfsong[3],
      'splits': tfsong[4],
       'r_wav': tfsong[5],
       'wspec': tfsong[6],
       'alpha': tfsong[7],
          'mu': tfsong[8],
      'sigma1': tfsong[9],
      'sigma2': tfsong[10],
       'vocal': tfsong[11],
      'smooth': tfsong[12],
         'snw': tfsong[13],
     's_alpha': tfsong[14],
        's_mu': tfsong[15],
    's_sigma1': tfsong[16],
    's_sigma2': tfsong[17],
          #'fs': tfsong[0],
        }

wf    = fsong['w'][:,0]

alpha = fsong['s_alpha']
beta  = fsong['s_mu']

def simulate(alpha, beta):
    simulate_for = alpha.shape[0]
    #simulate_for = alpha.shape[0] * 10
    x = np.zeros(simulate_for)
    v = np.zeros(simulate_for)
    x[0] = 0.0
    v[0] = 0.0
    dt = 1. / srate
    print dt
    gamma = 24000.00

    for t in range(1, simulate_for):
        st = t
        #st = t / 10
        xt = x[t-1]
        vt = v[t-1]
        vdot = alpha[st] * gamma ** 2 - beta[st] * gamma ** 2 * xt - \
               gamma ** 2 * xt ** 3 - gamma * xt ** 2 * vt + \
               gamma ** 2 * xt ** 2 - gamma * xt * vt
        v[t] = v[t-1] + vdot * dt
        x[t] = x[t-1] + v[t] * dt
    return x


wf = simulate(alpha, beta)
plt.plot(alpha)
plt.title("Alpha")
plt.show()
plt.plot(beta)
plt.title("Beta")
plt.show()
plt.plot(wf)
plt.title("Simulated Wave")
plt.show()

pdb.set_trace()

# Frames: 30870.00d, Rate: 44100.00

wl = 0.007 # 7ms
ic = 0.001 # 1ms

t,freq,timefreq,rms = gaussian_stft(wf, 44100, wl, ic)

spec = np.abs(timefreq)
spec = spec/spec.max()
nz = spec > 0
spec[nz] = 20*np.log10(spec[nz]) + 50
spec[spec < 0] = 0

plot_spectrogram(t, freq, spec)
plt.show()
