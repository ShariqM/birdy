import oscillators as os
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy.io as io
import numpy as np
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

#dt = 1. / srate
dt = 1e-6
ht = 5000 # hold time
duration = 5e-3

no = os.NormalOscillator()

#x = np.zeros(ht * alpha.shape[0])
#v = np.zeros(ht * alpha.shape[0])

def mike_simulate(ht):
    x,v = np.zeros(ht), np.zeros(ht)
    x[0], v[0] = 0.0, 0.0
    i = 0
    cparams = {'alpha':-0.41769, 'beta':0.346251775}
    r = no.run_simulation(cparams, duration, dt, x[i*ht], v[i*ht])
    x[i*ht:(i+1)*ht] = r[:,0]
    v[i*ht:(i+1)*ht] = r[:,1]
    return x

def simulate(ht):
    x,v = np.zeros(ht), np.zeros(ht)
    x[0], v[0] = 0.0, 0.0
    alpha, beta = (-0.41769, 0.346251775)
    gamma = 23500.0
    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        vdot = alpha * gamma ** 2 - beta * gamma ** 2 * xt - \
               gamma ** 2 * xt ** 3 - gamma * xt ** 2 * vt + \
               gamma ** 2 * xt ** 2 - gamma * xt * vt
        v[t] = v[t-1] + vdot * dt
        x[t] = x[t-1] + v[t] * dt
    return x

x  = mike_simulate(ht)
x2 = simulate(ht)

#plt.plot(x[3500:4000])
#plt.plot(x[14500:15500])
plt.subplot(121)
plt.title("Runge-Kutta Method")
plt.plot(x)
plt.subplot(122)
plt.title("Euler Method")
plt.plot(x2)
plt.show()
