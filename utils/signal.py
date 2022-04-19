from utils.base import *

def signal_power(signal):
  return np.mean(signal ** 2)

def power_to_db(power):
  return 10 * np.log10(power)

def calculateNoiseRatio(signal, noise, snr_db):
  s_power = signal_power(signal)
  n_power = signal_power(noise)
  ratio = (s_power / (n_power * (10 ** (snr_db / 10)))) ** 0.5
  return ratio