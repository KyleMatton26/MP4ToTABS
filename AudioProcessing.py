#Testing for MP4Proj

import wave
import numpy as np
import scipy 
from pydub import AudioSegment

#Storing the mp3 file
mp3_file = "HotCrossBuns.mp3"

#Opening the mp3 file as a WAV file and I THINK storing it
wav_file = AudioSegment.from_mp3(mp3_file)
