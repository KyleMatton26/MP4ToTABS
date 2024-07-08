import wave
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import medfilt
import os

# Storing the mp3 file
mp3_file = "HotCrossBuns.mp3"

# Check if the file exists
if not os.path.exists(mp3_file):
    raise FileNotFoundError(f"{mp3_file} does not exist.")

print("Loading MP3 file...")
# Get MP3 as audio segment
audio = AudioSegment.from_mp3(mp3_file)

# Convert the audio to mono
audio = audio.set_channels(1)

# Make the audio into a WAV file
wav_file = "HotCrossBuns.wav"
print("Converting to WAV...")
audio.export(wav_file, format="wav")

# Read the WAV file
samplerate, data = wavfile.read(wav_file)

# Using the documentation to plot the WAV file
def make_plot(samplerate, data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])

    plt.plot(time, data, label="Mono channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

# make_plot(samplerate, data)

#Number of samples in each frame - This typically is a power of 2 between 256-8192 (Stated in the video u posted)
frame_size = 2048
#Frame 1: 0-2048, Frame 2: 1024 - 3072. Hop size is the amount of samples you skip over to start the next frame. I chose 1024 because typically each frame should contain 50% overlap. Smaller hop size, more overlap. Larger hop size, less overlap.
hop_size = 1024

# Function to apply median filtering to smooth the frequency plot. Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html
def median_smooth(frequencies, kernel_size=5):
    smoothed_frequencies = medfilt(frequencies, kernel_size)
    return smoothed_frequencies

#For this method, im using rfft and rfftfreq instead of fft and fftfreq because we are dealing with real values, typically done with audio.
def get_domiant_frequency(frame, samplerate):
    #Converts "frame" from a time-domain signal to a frequency domain signal
    fft_signal = rfft(frame)

    #Gets the magnitude of each frequency of the wave
    fft_signal_abs = np.abs(fft_signal)
    print("Length of fft_signal_abs:", len(fft_signal_abs))
    print(fft_signal_abs)

    # Converts the magnitude of each frequency to its frequency in Hz. Documentation for rfftfreq: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html
    frequencies = rfftfreq(len(frame), 1 / samplerate)

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

# Array to store the frequencies from the audio file
frequencies = []
times = []

# Table with expected Hz values for the notes (C4, D4, E4)
expected_hz_values = {
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63
}

# Loop through each frame and add the dominant frequency to the array
for i in range(number_of_frames):
    start = i * hop_size
    end = start + frame_size
    frame = data[start:end]
    
    if len(frame) != frame_size:
        print(f"Skipping frame {i} due to incorrect size: {len(frame)}")
        continue
        #Note: This is skipping the last frame entirely. Will add padding to last frame to be processed at a later time
    
    dominant_frequency = get_domiant_frequency(frame, samplerate)
    frequencies.append(dominant_frequency)
    
    # Calculate time for current frame
    current_time = start / samplerate
    times.append(current_time)

# Apply median filtering to smooth the frequency plot
smoothed_frequencies = median_smooth(frequencies, kernel_size=5)  # Adjust kernel_size as needed: Larger kernel size means a larger window to take the median of. Would miss small changes but would be more accurate for longer notes if kernel size is bigger

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(times, smoothed_frequencies, label="Smoothed Dominant Frequency")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Smoothed Dominant Frequency Over Time")
plt.legend()

# Add expected Hz values as horizontal lines for C4, D4, E4 notes (Thank you ChatGPT :) )
for note, hz_value in expected_hz_values.items():
    plt.axhline(y=hz_value, color='r', linestyle='--', label=f'{note} (Expected {hz_value} Hz)')

plt.legend()
plt.grid(True)
plt.show()
