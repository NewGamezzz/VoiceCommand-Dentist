from utils.base import *

def plot_spectrogram(audio):
  S = np.abs(librosa.stft(audio))
  fig, ax = plt.subplots(figsize=(20, 10))
  img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                        ref=np.max),
                                y_axis='log', x_axis='time', ax=ax)
  ax.set_title('Power spectrogram')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")

def cut_audio(audio_path, out_path, second=10, verbose=False):
  filename = os.path.basename(audio_path).split('.')[0]
  audio, sr = librosa.load(audio_path)

  buffer = second * sr

  samples_total = len(audio)
  samples_wrote = 0

  print("Audio Filepath:", audio_path)
  print("Length:", len(audio)/sr, 'second')

  out_file = []
  pbar = tqdm(range(buffer, samples_total + buffer, buffer), desc="Cutting Process")
  for idx, stop in enumerate(pbar):
    start = samples_wrote
    window = audio[start:stop]
    
    out_file.append(os.path.join(out_path, filename + '_' + str(idx) + '.wav'))
    wavfile.write(out_file[-1], sr, window.astype('f4'))
    if verbose:
      print(f"Start:{start/sr} Stop: {stop/sr}")
    samples_wrote = stop

  return out_file

def segmentAudio(audio, sr, second):
  step = int(second * sr) # <== change from second to step
  audio_length = len(audio)
  start = 0

  audio_li = []
  for stop in range(step, audio_length + step, step):
    cut_audio = audio[start:stop]
    audio_li.append(cut_audio)
    start = stop
  return np.array(audio_li, dtype=object)

def peakLevelAudio(audio, sr, second):
  norm_audio = np.abs(audio)
  li = segmentAudio(norm_audio, sr, second)
  l = np.array(list(map(lambda x: max(x), li)))
  return l

def SpectralGate(audio_path, out_path, increase_volume=False, config={}):
  filename = os.path.basename(audio_path).split('.')[0]
  audio, sr = librosa.load(audio_path)
  reduced_noise = nr.reduce_noise(y=audio, sr=sr, **config)
  if increase_volume:
    reduced_noise *= 3
    reduced_noise = np.clip(reduced_noise, -1., 1.)
  wavfile.write(os.path.join(out_path, filename + '.wav'), sr, reduced_noise.astype('f4')) # <= previous work multiply by 3 before save

def Speech2TextAPI(
  filepath,
  base_data = {
      "audioData":None,
      "decoder_type": "BeamSearch",
      "get_word_timestamps": True,
  },
  header = {
      'Content-type': 'application/json',
      'x-api-key':'8yYrTF/Ya97OBpRaKok9+pBeBjkXK6Wkl0j4dPHdrCs='
  }):

  # filepath = f'shortAudio/{mode}/{filename}/{filename}_{idx}.wav'
  with open(filepath, 'rb') as fh:
    content = fh.read()

  base_data['audioData'] = base64.encodebytes(content).decode('utf-8')
  res = requests.post(
    'https://gowajee-api-mgmt.azure-api.net/speech/transcribe',
    data=json.dumps(base_data),
    headers=header
  )

  return res.json()
