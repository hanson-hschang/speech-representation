import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

train_audio_path = 'speech_commands'
audio_graph_path = 'speech_graph'

# keywords = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
# keywords = ['cat']

keywords = ['bed', 'bird', 'dog', 'down', 'eight', 'five', 'four', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'three', 'tree', 'two', 'up', 'wow', 'zero']

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

fig = plt.figure(figsize=(14, 8))

for keyword in keywords:

	filenames = os.listdir(train_audio_path + '/' +keyword)
	filenames = [os.path.splitext(filename)[0] for filename in filenames]
	
	# filenames = ['0a7c2a8d_nohash_0','0a9f9af7_nohash_0', '0a9f9af7_nohash_1', '0a9f9af7_nohash_2']
	# filenames = ['0a7c2a8d_nohash_0']

	for filename in filenames:
		filename = '/' + keyword + '/' + filename
		sample_rate, samples = wavfile.read(str(train_audio_path) + filename + '.wav')
		mean = np.mean(samples)
		std = np.std(samples)
		samples = (samples - mean) / std
		
		freqs, times, spectrogram = log_specgram(samples, sample_rate)
		mean = np.mean(spectrogram, axis=0)
		# std = np.std(spectrogram, axis=0)
		# spectrogram = (spectrogram - mean) / std
		maxium = np.amax(np.absolute(spectrogram))
		spectrogram = (spectrogram - mean) / maxium
		
		fig.clf()
		ax1 = fig.add_subplot(211)
		ax1.set_title('Raw wave of ' + filename)
		ax1.set_ylabel('Amplitude')
		ax1.set_xlim([-0.01, 1.01])
		ax1.set_ylim([-10.01, 10.01])
		ax1.plot(np.linspace(0, len(samples)/sample_rate, len(samples)), samples)

		ax2 = fig.add_subplot(212)
		ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
			       extent=[times.min(), times.max(), freqs.min(), freqs.max()], vmin=-1.2, vmax=1.2,)
		ax2.set_xlim(ax1.get_xlim())
		ax2.set_yticks(freqs[::16])
		ax2.set_title('Spectrogram of ' + filename)
		ax2.set_ylabel('Freqs in Hz')
		ax2.set_xlabel('Seconds')
		
		ax3 = fig.add_axes([0.92, 0.09, 0.03, 0.4])
		mappable = ax2.images[0]
		plt.colorbar(mappable=mappable, cax=ax3, ticks=np.linspace(-1.2, 1.2, 9))
		
		if not os.path.isdir(audio_graph_path + '/' + keyword):
			os.makedirs(audio_graph_path + '/' + keyword)
		fig.savefig(audio_graph_path + filename +'.png')
