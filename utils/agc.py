import numpy as np
class Expect:
  def expectation(self, mu, sigma, data):
    zb = (data - mu[0]) / sigma[0]
    zs = (data - mu[1]) / sigma[1]
    return (zs**2 < zb**2).astype(int) 

class Maximize:
  def maximization(self, mu, sigma, k_mu, k_sigma, expect, data):
    new_mu = mu.copy()
    new_mu[expect] = k_mu * data + (1-k_mu) * mu[expect]
    new_var = sigma.copy() ** 2
    new_var[expect] = k_sigma * ((data-new_mu[expect])**2) + (1-k_sigma) * new_var[expect]
    new_sigma = new_var ** 0.5
    return new_mu, new_sigma

class AGCModel:
  def __init__(self, expect=None, maximize=None, k_mu=0.5, k_sigma=0.33, theta0=0.8, theta1=0.1):
    self.expect = expect
    self.maximize = maximize
    self.k_mu = k_mu
    self.k_sigma = k_sigma
    self.theta0 = theta0
    self.theta1 = theta1
    self.t = 0
    self.mu = None
    self.sigma = None
    # self.init_param()

  def init_param(self, data):
    # self.mu = np.zeros(2)
    # self.sigma = np.ones(2) * 0.25
    self.mu = np.ones(2) * data.mean()
    self.sigma = np.ones(2) * data.std()

  def expectation(self, mu, sigma, data):
    return self.expect.expectation(mu, sigma, data)
  
  def maximization(self, mu, sigma, k_mu, k_sigma, expect, data):
    return self.maximize.maximization(mu, sigma, k_mu, k_sigma, expect, data)
    
  def update(self, t, delta, new_mu, new_sigma):
    new_var = new_sigma ** 2
    if np.any(new_var < t**2):
      new_var += new_var.sum() / (2 * delta)

    self.mu = new_mu
    self.sigma = new_var ** 0.5
    # Sort mu, so that class s is at index 1
    mask = np.argsort(self.mu)
    self.mu = self.mu[mask]
    self.sigma = self.sigma[mask]
  
  def computeGain(self, expect):
    if expect == 0: # <= predict as background
      return 1
    mu = self.mu
    sigma = self.sigma
    gain = 1
    if mu[1] - mu[0] > sigma[1] + sigma[0]:
      gain  = self.theta0 / (mu[1] + sigma[1])
    else:
      gain = self.theta1 / min(mu[1] + sigma[1], mu[0] + sigma[0])
    return max(1, gain) # <= not attenuate

  def run(self, data, delta=16):
    if self.mu is None or self.sigma is None:
      self.init_param(data)
    t = (0.5 * max(data))**2
    if t > self.t:
      self.t = t

    gain = []
    for l in data:
      expect = self.expectation(self.mu, self.sigma, l) # <= Expectation
      gain.append(self.computeGain(expect)) # <= Compute Gain
      new_mu, new_sigma = self.maximization(self.mu, self.sigma, self.k_mu, self.k_sigma, expect, l) # <= Maximization
      self.update(self.t, delta, new_mu, new_sigma) # <= Update (decay)
    return np.array(gain)

  def scaleUp(self, gain, audio_li):
    max_func = lambda x: abs(max(x))
    min_func = lambda x: abs(min(x))

    max_amp, min_amp = np.array(list(map(max_func, audio_li))).reshape(-1, 1), np.array(list(map(min_func, audio_li))).reshape(-1, 1)
    peak_amp = np.concatenate((max_amp, min_amp), axis=1).max(1)
    peak_amp_gain = peak_amp * gain

    mask = peak_amp_gain > 1
    gain_no_clip = gain.copy()
    gain_no_clip[mask] = 1 / peak_amp[mask]
    gain_no_clip[gain_no_clip < 1] = 1 # <= No attenuate
    up_scale_audio_li = audio_li.copy()
    if len(up_scale_audio_li.shape) == 2:
      gain_no_clip = np.expand_dims(gain_no_clip, axis=1)
    up_scale_audio_li *= gain_no_clip
    return up_scale_audio_li

  def set_expect(self, expect):
    self.expect = expect
  
  def set_maximize(self, maximize):
    self.maximize = maximize