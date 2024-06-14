#Testing for MP4Proj

import wave
import numpy as np
from scipy.io import wavfile 
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import rfft,rfftfreq

#Storing the mp3 file
mp3_file = "HotCrossBuns.mp3"

#Get MP3 as audio segment 
audio = AudioSegment.from_mp3(mp3_file)

#Make the audio into a WAV file
wav_file = "HotCrossBuns.wav"
audio.export(wav_file, format="wav")


#Read the WAV file
samplerate, data = wavfile.read(wav_file)

#Using the documentation to plot the WAV file
def make_plot(samplerate, data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])

    plt.plot(time, data[:, 0], label="Left channel")
    plt.plot(time, data[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

#make_plot(samplerate, data)

#Number of samples in each frame - This typically is a power of 2 between 256-8192 (Stated in the video u posted)
frame_size = 2048 
#Frame 1: 0-2048, Frame 2: 1024 - 3072. Hop size is the amount of samples you skip over to start the next frame. I chose 1024 because typically each frame should contain 50% overlap. Smaller hop size, more overlap. Larger hop size, less overlap.
hop_size = 1024

#For this method, im using rfft and rfftfreq instead of fft and fftfreq because we are dealing with real values, typically done with audio.
def get_domiant_frequency(frame, samplerate):
    #Converts "frame" from a time-domain signal to a frequency domain signal
    fft_signal = rfft(frame) 

    #Gets the magnitude of each frequency of the wave 
    fft_signal_abs = np.abs(fft_signal)
    print("Length of fft_signal_abs:", len(fft_signal_abs))
    print(fft_signal_abs)
    # Converts the magnitude of each frequency to its frequency in Hz. Documentation for rfftfreq: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html
    frequencies = rfftfreq(len(fft_signal), 1 / samplerate)

    #Get the dominant frequency by using argmax to find the index of the frequency with the strongest magnitude 
    dominant_frequency_index = np.argmax(fft_signal_abs)
    print(dominant_frequency_index)
    #uses the index to get the frequency in Hz
    dominant_frequency = frequencies[dominant_frequency_index]

    #Returns the dominant frequency 
    return dominant_frequency

#(len(data) - frame_size): This is done because this tells us the amount of full frames we will be processing. The last frame will start at (len(data) - frame_size) and continue until the end.
# // hop_size: This is used to see how many hops will go into (len(data) - frame_size). This determines the number of frames that will go into our audio data
#We + 1 because we have to include the initial frame before the first hop
number_of_frames = ((len(data) - frame_size) // hop_size) + 1 

#Array to store the frequencies from the audio file
frequencies = []

#Loop through each frame and add the dominant frequency to the array
for i in range(number_of_frames):
    start = i * hop_size 
    end = start + frame_size
    frame = data[start:end]
    dominant_frequency = get_domiant_frequency(frame, samplerate)
    frequencies.append(dominant_frequency)

start = number_of_frames * hop_size

#Get the the dominant frequency from the last partial frame
if start < (len(data)):
    frame = data[start:]
    #Pad the last frame with zeros so it matches frame size
    if len(frame) < frame_size:
        pad_length = frame_size - len(frame)
        frame = np.pad(frame, (0, pad_length), "constant")
    #Trim the last fram so it matches frame size
    elif len(frame) > frame_size:
        frame = frame[:frame_size]
    dominant_frequency = get_domiant_frequency(frame, samplerate)
    frequencies.append(dominant_frequency)

#Gets an array of the times where each frame starts
times = (np.arange(len(frequencies)) * hop_size) / samplerate
plt.figure(figsize=(10, 4))
plt.plot(times, frequencies)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Dominant Frequency Over Time")
plt.show()