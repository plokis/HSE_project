import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.optimize
from scipy.signal import hilbert
from scipy.signal import argrelextrema
from sphinx.addnodes import index

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def manual_convolution(array, kernel):
    array_len = len(array)
    kernel_len = len(kernel)
    pad_width = kernel_len // 2
    padded_array = np.pad(array, pad_width, mode='edge')  # Паддинг с повторением краёв
    result = np.zeros_like(array)

    for j in range(array_len):
        result[j] = np.sum(padded_array[j:j + kernel_len] * kernel)

    return result

def pantelleev_filter(t, omega_0):
    """
    Pulse response of Panteleev filter.

    Parameters:
    t (array): Initial dates.
    omega_0 (int/float): half-width parameter ω0

    Returns:
    array:
        - Impulse response of the filter
    """
    coef = omega_0 / (2 * np.sqrt(2))
    exp_part = np.exp(-omega_0 * np.abs(t) / np.sqrt(2))
    cos_part = np.cos(omega_0 * t / np.sqrt(2))
    sin_part = np.sin(omega_0 * np.abs(t) / np.sqrt(2))
    return coef * exp_part * (cos_part + sin_part)

def fit_sin(tt, yy):
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def ampl_fft(signal, dT):
    N = len(signal)
    spectrum = fft(signal, n=N) / N
    omega = fftfreq(N, dT)
    return spectrum, omega

def split_data(data):
    # data = data.tolist()
    s01, s02 = [], []
    chunk01 = data[:77]
    s01 = np.concatenate([s01,chunk01])
    for i in range(len(data) // 365 + 100):
        chunk1 = data[i * 365 + 265:i * 365 + 265 + 100 + 77]
        s01 = np.concatenate([s01,chunk1])
    for i in range(len(data) // 365):
        chunk2 = data[i * 365 + 77:i * 365 + 265]
        s02 = np.concatenate([s02,chunk2])
    s1 = np.array(s01)
    s2 = np.array(s02)
    return s1, s2

df_rad = pd.read_csv('dataexport_20241204T120753.csv',sep=',')
df_tide = pd.read_csv('LOD_zonaltide.dat',sep='\s+')

dates = np.array(df_rad['Date'])
rad_val = np.array(df_rad['rad_val'])

dates_tide = np.array(df_tide['year'])
tide_1 = np.array(df_tide['dut'])[np.where(dates_tide == 1940.)[0][0]:np.where(dates_tide == 2024.88191)[0][0]]
tide_2 = np.array(df_tide['dlod'])[np.where(dates_tide == 1940.)[0][0]:np.where(dates_tide == 2024.88191)[0][0]]
tide_3 = np.array(df_tide['omega'])[np.where(dates_tide == 1940.)[0][0]:np.where(dates_tide == 2024.88191)[0][0]]

for i in range(len(dates)):
    dates[i] = dates[i].replace('T0000', '')

print(np.where(dates == '19400319')[0], np.where(dates == '19400923')[0])

dt = 1
k = np.arange(0, len(dates))

sine_tide = fit_sin(k, tide_2)['fitfunc']

res = fit_sin(k, rad_val)
print(res)

detrended_rad = rad_val - res['fitfunc'](k)

print(len(detrended_rad))

specter_rad, omega_rad = ampl_fft(rad_val, dt)
detrend_specter_rad, detrend_omega_rad = ampl_fft(detrended_rad, dt)

autocor = np.correlate(detrended_rad, detrended_rad, 'same') / (len(detrended_rad)*len(detrended_rad))

# 1. Два суб ряда
winter_dates, summer_dates = split_data(dates)
winter_data, summer_data = split_data(detrended_rad)

winter_specter, winter_omega = ampl_fft(winter_data, dt)
summer_specter, summer_omega = ampl_fft(summer_data, dt)
print(len(winter_data), len(summer_data))

# 2. Огибающая
omega_0 = 1
t = np.arange(0, 200)
h = pantelleev_filter(t, omega_0)
max_indices = argrelextrema(detrended_rad, np.greater, order=10)[0]
min_indices = argrelextrema(detrended_rad, np.less, order=10)[0]
max_indices_abs = argrelextrema(np.abs(detrended_rad), np.greater, order=5)[0]
abs_envelope = np.interp(k, k[max_indices_abs], np.abs(detrended_rad)[max_indices_abs])
abs_envelope_conv = np.convolve(abs_envelope, h, 'same')
upper_envelope = np.interp(k, k[max_indices], detrended_rad[max_indices])
lower_envelope = np.interp(k, k[min_indices], detrended_rad[min_indices])

sine_envelope_abs = fit_sin(k, abs_envelope)['fitfunc']

# detrended_deenveloped_rad = detrended_rad - upper_envelope - lower_envelope
print(len(upper_envelope), detrended_rad)

plt.figure(1)
plt.plot(dates, df_rad['rad_val'])
plt.plot(k, res['fitfunc'](k))
plt.xticks([dates[60*i] for i in range(517)])
plt.xlabel('Даты')
plt.ylabel('Мера солнечного излучения, Вт/м2')
plt.grid()

plt.figure(2)
plt.plot(dates, detrended_rad)
plt.plot(10000*(tide_2 - sine_tide(k)))
plt.xticks([dates[60*i] for i in range(517)])
plt.title('')
plt.grid()

plt.figure(3)
plt.plot(res['period']*omega_rad[1:len(omega_rad)//2], np.abs(specter_rad[1:len(specter_rad)//2]))
plt.grid()

plt.figure(4)
plt.plot(1/(omega_rad), np.abs(specter_rad))
plt.xscale("log")
plt.grid()

plt.figure(5)
plt.plot(res['period']*detrend_omega_rad[:len(omega_rad)//2], np.abs(detrend_specter_rad[:len(specter_rad)//2]))
plt.grid()

plt.figure(6)
plt.plot(1/(detrend_omega_rad), np.abs(detrend_specter_rad))
plt.xscale("log")
plt.grid()

plt.figure(7)
plt.plot(autocor[len(autocor)//2:], linewidth=0.75)
plt.grid()

plt.figure(8)
plt.plot(winter_dates, winter_data, linewidth=0.75)
plt.xticks([winter_dates[60*i] for i in range(250)])
plt.title('"Зимний" суб ряд, с небольшим разбросом (между 23 сентября и 19 марта)')
plt.xlabel('Даты')
plt.grid()

plt.figure(9)
plt.plot(summer_dates, summer_data, linewidth=0.75)
plt.xticks([summer_dates[60*i] for i in range(264)])
plt.title('"Летний" суб ряд, с большим разбросом (между 19 марта и 23 сентября)')
plt.grid()

plt.figure(10)
plt.plot(res['period']*winter_omega[:len(winter_omega)//2], np.abs(winter_specter[:len(winter_omega)//2]), linewidth=0.75)
plt.title('Спектр "Зимнего" суб ряда')
plt.grid()

plt.figure(11)
plt.plot(res['period']*summer_omega[:len(summer_omega)//2], np.abs(summer_specter[:len(summer_omega)//2]), linewidth=0.75)
plt.title('Спектр "Летнего" суб ряда')
plt.grid()

plt.figure(12)
plt.plot(dates, np.abs(detrended_rad), label='Ряд без годового тренда')
plt.plot(k, abs_envelope, label='Верхняя огибающая')
# plt.plot(dates, lower_envelope, label='Нижняя огибающая')
plt.legend(loc='best')
plt.title('Данные вместе с огибающими')
plt.grid()
#
# plt.figure(13)
# plt.plot(dates, np.abs(detrended_rad) - abs_envelope)
# plt.title('Данные без годового тренда и без верхней огибающей')
# plt.grid()
#
# plt.figure(14)
# plt.plot(dates, detrended_rad - lower_envelope)
# plt.title('Данные без годового тренда и без нижней огибающей')
# plt.grid()
#
# plt.figure(15)
# plt.plot(dates, np.abs(detrended_rad))
# plt.plot(abs_envelope)
# plt.grid()

plt.show()

