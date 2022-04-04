import librosa
import playsound
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import noisereduce as nr
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from scipy.signal import spectrogram

import requests
import json
import base64

import warnings
## Librosa load can't read mp3, therefore it will call audioread instead.
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
rng = np.random.default_rng() 