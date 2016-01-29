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
    return x, v

def get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3):
    return alpha * gamma ** is_2 + beta * gamma ** is_2 * xt - \
           gamma ** is_2 * xt ** is_3 - gamma * xt ** is_2 * vt + \
           gamma ** is_2 * xt ** is_2 - gamma * xt * vt

def simulate(ht):
    x,v = np.zeros(ht, dtype='float64'), np.zeros(ht, dtype='float64')
    x[0], v[0] = np.float64(0.0), np.float64(0.0)
    is_2 = np.float64(2.0)
    is_3 = np.float64(3.0)
    alpha, beta = (np.float64(-0.41769), np.float64(0.346251775))
    gamma = np.float64(23500.0)
    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        vdot = dt * get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3)
        v[t] = v[t-1] + vdot * dt
        x[t] = x[t-1] + v[t] * dt
    return x, v

def simulate_rk2e(ht):
    x,v = np.zeros(ht), np.zeros(ht)
    x[0], v[0] = 0.0, 0.0
    is_2 = 2.0
    is_3 = 3.0
    alpha, beta = -0.41769, 0.346251775
    gamma = 23500.0

    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        k1 = dt * get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3)
        k2 = dt * get_vdot(xt, vt + (1/2.) * k1, alpha, beta, gamma, is_2, is_3)
        v[t] = v[t-1] + k2
        x[t] = x[t-1] + v[t] * dt
    return x, v


def simulate_rk2e_64(ht):
    #x,v = np.zeros(ht, dtype='float64'), np.zeros(ht, dtype='float64')
    x,v = np.zeros(ht), np.zeros(ht)
    x[0], v[0] = np.float64(0.0), np.float64(0.0)
    x[0], v[0] = np.float64(0.0), np.float64(0.0)
    is_2 = np.float64(2.0)
    is_3 = np.float64(3.0)
    alpha, beta = (np.float64(-0.41769), np.float64(0.346251775))
    gamma = np.float64(23500.0)

    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        k1 = dt * get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3)
        k2 = dt * get_vdot(xt, vt + (1/2.) * k1, alpha, beta, gamma, is_2, is_3)
        v[t] = v[t-1] + k2
        x[t] = x[t-1] + v[t] * dt
    return x, v

def simulate_rk4e(ht):
    x,v = np.zeros(ht, dtype='float64'), np.zeros(ht, dtype='float64')
    x[0], v[0] = np.float64(0.0), np.float64(0.0)
    is_2 = np.float64(2.0)
    is_3 = np.float64(3.0)
    alpha, beta = (np.float64(-0.41769), np.float64(0.346251775))
    gamma = np.float64(23500.0)

    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        k1 = dt * get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3)
        k2 = dt * get_vdot(xt, vt + (1/2.) * k1, alpha, beta, gamma, is_2, is_3)
        k3 = dt * get_vdot(xt, vt + (1/2.) * k2, alpha, beta, gamma, is_2, is_3)
        k4 = dt * get_vdot(xt, vt + k3, alpha, beta, gamma, is_2, is_3)
        v[t] = v[t-1] + (1/6.) * k1 + (1/3.) * k2 + (1/3.) * k3 + (1/6.) * k4
        x[t] = x[t-1] + v[t] * dt
    return x, v

def simulate_bogshamp(ht):
    x,v = np.zeros(ht, dtype='float64'), np.zeros(ht, dtype='float64')
    x[0], v[0] = np.float64(0.0), np.float64(0.0)
    is_2 = np.float64(2.0)
    is_3 = np.float64(3.0)
    alpha, beta = (np.float64(-0.41769), np.float64(0.346251775))
    gamma = np.float64(23500.0)

    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        k1 = dt * get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3)
        k2 = dt * get_vdot(xt, vt + (1/2.) * k1, alpha, beta, gamma, is_2, is_3)
        k3 = dt * get_vdot(xt, vt + (3/4.) * k2, alpha, beta, gamma, is_2, is_3)
        k4 = dt * get_vdot(xt, vt + (7/24.) * k1 + (1/3.) * k2 + (4/9.) * k3, alpha, beta, gamma, is_2, is_3)
        v[t] = v[t-1] + (1/24.) * k1 + (1/4.) * k2 + (1/3.) * k3 + (1/8.) * k4
        x[t] = x[t-1] + v[t] * dt
    return x, v

def simulate_rkck(ht):
    x,v = np.zeros(ht, dtype='float64'), np.zeros(ht, dtype='float64')
    x[0], v[0] = np.float64(0.0), np.float64(0.0)
    is_2 = np.float64(2.0)
    is_3 = np.float64(3.0)
    alpha, beta = (np.float64(-0.41769), np.float64(0.346251775))
    gamma = np.float64(23500.0)

    for t in range(1, ht):
        xt = x[t-1]
        vt = v[t-1]
        k1 = dt * get_vdot(xt, vt, alpha, beta, gamma, is_2, is_3)
        k2 = dt * get_vdot(xt, vt + (1/5.) * k1, alpha, beta, gamma, is_2, is_3)
        k3 = dt * get_vdot(xt, vt + (3/40.) * k1 + (9/40.) * k2, alpha, beta, gamma, is_2, is_3)
        k4 = dt * get_vdot(xt, vt + (3/10.) * k1 - (9/10.) * k2 + (6/5.) * k3, alpha, beta, gamma, is_2, is_3)
        k5 = dt * get_vdot(xt, vt - (11/54.) * k1 + (5/2.) * k2 - (70/27.) * k3 + (35/27.) * k4, alpha, beta, gamma, is_2, is_3)
        k6 = dt * get_vdot(xt, vt + (1631/55296.)    * k1 \
                                  + (175/512.)       * k2 \
                                  + (575/13824.)     * k3 \
                                  + (44275/1100592.) * k4 \
                                  + (253/4096.)      * k5, alpha, beta, gamma, is_2, is_3)
        v[t] = v[t-1] + (2825/27648.)  * k1 \
                      + (18575/48384.) * k3 \
                      + (13525/55296.) * k4 \
                      + (277/14336.)   * k5 \
                      + (1/4.)         * k6

        x[t] = x[t-1] + v[t] * dt
    return x,v


#x,v    = mike_simulate(ht)
x2, v2 = simulate(ht)
x3, v3 = simulate_rk2e(ht)

#plt.plot(x[3500:4000])
#plt.plot(x[14500:15500])
#plt.subplot(231)
#plt.title("x (C method) ")
#plt.plot(x)
#plt.subplot(234)
#plt.title("v (C method) ")
#plt.plot(v)
plt.subplot(234)
plt.title("Spectrum (C method) ")
plt.plot(np.fft.fftfreq(len(x), 1./dt), np.fft.fft(x) ** 2)


plt.subplot(232)
plt.title("x2 (Euler Method)")
plt.plot(x2)
plt.subplot(235)
plt.title("v2 (Euler Method)")
plt.plot(v2)

plt.subplot(233)
plt.title("x3 (Test Method)")
plt.plot(x3)
plt.subplot(236)
plt.title("v3 (Test Method)")
plt.plot(v3)


plt.show()
