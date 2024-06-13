#Testing for MP4Proj

import wave
import numpy as np
from scipy.io import wavfile 
from pydub import AudioSegment
import matplotlib as plt

#Storing the mp3 file
mp3_file = "HotCrossBuns.mp3"

#Opening the mp3 file as a WAV file and I THINK storing it
wav_file = AudioSegment.from_mp3(mp3_file)

#Read the WAV file
samplerate, data = wavfile.read(wav_file)

#Using the documentation to plot the WAV file

length = data.shape[0] / samplerate
time = np.linspace(0., length, data.shape[0])

plt.plot(time, data[:, 0], label="Left channel")
plt.plot(time, data[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()