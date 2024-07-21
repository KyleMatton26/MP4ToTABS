import wave
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import medfilt
import os
import librosa

# Storing the mp3 file
mp3_file = "HotCrossBuns.mp3"
#mp3_file = "TwinkleTwinkleLittleStar.mp3"

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
#wav_file = "TwinkleTwinkleLittleStar.wav"
print("Converting to WAV...")
audio.export(wav_file, format="wav")


# Load the audio file
y, sr = librosa.load(wav_file)

# Onset detection
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
print("Detected onsets at times:", onset_times)
first_onset_time = onset_times[0] - 0.06657596 #Added the "- 0.06657596" because that was the discrepency on the 120BPM Test

def get_onset_frames():
    return onset_frames

#Chunk to get the tempo
# Parameters
hop_length = 256  # Adjust as needed
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
# Refine onset envelope
onset_env = librosa.util.normalize(onset_env)
# Extract the tempo using the refined onset envelope
tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
tempo = int(tempo)
print(f"Estimated tempo: {tempo} BPM")


def get_tempo():
    return tempo

#NEED TO TAKE THE TEMPO AND USE IT WITH THE SAMPLE RATE TO GET HOW MANY SAMPLES ARE IN A 32nd NOTE
#SET THAT AS THE FRAME SIZE THEN MAKE HOP SIZE HALF THAT

# Read the WAV file
samplerate, data = wavfile.read(wav_file)

print(f"Estimated sample rate: {samplerate} SPS")

# Using the documentation to plot the WAV file
def make_plot(samplerate, data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])

    plt.plot(time, data, label="Mono channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    
#make_plot(samplerate, data)

what_type_of_note_modifier = 1 #1 - Quarter : 2 - Eigth : 4 - 16th : 8 - 32nd
beats_per_second = tempo/60
quarter_note_duration = 1/(beats_per_second*what_type_of_note_modifier) #*2 for 8th note : *4 for 16th note : *1 for quarter note
print(f"Estimated quarter note duration: {quarter_note_duration} seconds")
#Number of samples in each frame - This typically is a power of 2 between 256-8192 (Stated in the video u posted)
frame_size = int(quarter_note_duration*samplerate) #2048 was old frame size
#Frame 1: 0-2048, Frame 2: 1024 - 3072. Hop size is the amount of samples you skip over to start the next frame. I chose 1024 because typically each frame should contain 50% overlap. Smaller hop size, more overlap. Larger hop size, less overlap.
hop_size = int(frame_size/2) #1024 was old hop size

print(f"Estimated frame size: {frame_size} samples per frame")

def get_frame_size():
    return frame_size

def get_samplerate():
    return samplerate

def get_hop_length():
    return hop_length

# Compute and print the frame duration in seconds
frame_duration = hop_size / samplerate
print(f"Frame duration: {frame_duration} seconds")

def get_frame_duration():
    return frame_duration

# Function to apply median filtering to smooth the frequency plot. Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html
def median_smooth(frequencies, kernel_size=15):
    smoothed_frequencies = medfilt(frequencies, kernel_size)
    return smoothed_frequencies

#For this method, im using rfft and rfftfreq instead of fft and fftfreq because we are dealing with real values, typically done with audio.
def get_domiant_frequency(frame, samplerate):
    #Converts "frame" from a time-domain signal to a frequency domain signal
    fft_signal = rfft(frame)

    #Gets the magnitude of each frequency of the wave
    fft_signal_abs = np.abs(fft_signal) #Possibly change to it being squared, Quantum Physics Wave Functions and probibility

    # Converts the magnitude of each frequency to its frequency in Hz. Documentation for rfftfreq: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html
    frequencies = rfftfreq(len(frame), 1 / samplerate)

    #Get the dominant frequency by using argmax to find the index of the frequency with the strongest magnitude 
    dominant_frequency_index = np.argmax(fft_signal_abs)

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
    #HCB expected values
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63,
    
    #TTLS expected values
    #'C4': 261.63,
    #'D4': 293.66,
    #'E4': 329.63,
    #'G4': 392.00
    #'A4': 440.00,
    #'F4': 349.23
    
    
    
    #'C5': 523.25,
    #'G5': 783.99,
    #'A5': 880.00,
    #'F5': 698.46,
    #'D5': 587.33,
    #'E5': 659.26
}

# Loop through each frame and add the dominant frequency to the array
for i in range(number_of_frames):
    #start = i * hop_size + int(first_onset_time * sr) #off sets the starting frame to start on the first onset
    start = i * hop_size + (int(first_onset_time * sr) % hop_size) #Suggested by GPT, need to think if it's actually good
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
#kernel_size_test=int(frame_size/6000)*2+1 #Attempt to make kernel size based on the frame size
kernel_size_test = 1
smoothed_frequencies = median_smooth(frequencies, kernel_size=kernel_size_test)#=int(frame_size/6000)*2+1)#15 was previous  # Adjust kernel_size as needed: Larger kernel size means a larger window to take the median of. Would miss small changes but would be more accurate for longer notes if kernel size is bigger
print(f"Kernel size used for medfilt: {kernel_size_test}")
# Save the smoothed frequencies and times to a file for further analysis 
#Maybe try arrays later
np.savez("dominant_frequencies.npz", times=times, frequencies=smoothed_frequencies)

# Plotting
def make_smoothed_dominant_frequency_graph(times, smoothed_frequencies, expected_hz_values):
    plt.figure(figsize=(12, 6))
    plt.plot(times, smoothed_frequencies, label="Smoothed Dominant Frequency")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Smoothed Dominant Frequency Over Time")
    plt.legend()

    # Add expected Hz values as horizontal lines for C4, D4, E4 notes (Thank you ChatGPT :) )
    for note, hz_value in expected_hz_values.items():
        plt.axhline(y=hz_value, color='r', linestyle='--', label=f'{note} (Expected {hz_value} Hz)')
    """    
    # Add vertical lines for frame windows
    for i in range(number_of_frames):
        frame_start_time = (i * hop_size) / samplerate
        plt.axvline(x=frame_start_time, color='b', linestyle='--', alpha=0.5)
        
        
    """
    frame_counter = 1
    for i in range(int(number_of_frames)):
        frame_start_time = (i * hop_size/samplerate)
    #for i in range(int(number_of_frames / what_type_of_note_modifier / 2)):
        #frame_start_time = (i * hop_size * what_type_of_note_modifier * 2) / samplerate
        if (frame_counter % 2 == 0):
            plt.axvline(x=frame_start_time + first_onset_time, color='b', linestyle='--', alpha=0.5)
        else:
            plt.axvline(x=frame_start_time + first_onset_time, color='c', linestyle='--', alpha=0.5)
        frame_counter += 1
    
    
    #plt.axvline(x=first_onset_time, color='b', linestyle='--', alpha=0.5)
    #plt.axvline(x=first_onset_time+hop_size*16/sr, color='b', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.grid(True)
    plt.show()



print(f"Sample Per Second: {samplerate}")
print(f"Samples Per Frame: {frame_size}")
print(f"Seconds Per Frame: {frame_size / samplerate}")
print(f"How many frames we have: {number_of_frames}")
#print(f"Sample Rate: {samplerate}")


#uncomment this to see the graph
#make_smoothed_dominant_frequency_graph(times, smoothed_frequencies, expected_hz_values)
