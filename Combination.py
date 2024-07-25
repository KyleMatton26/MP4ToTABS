import librosa
import matplotlib.pyplot as plt 
import numpy as np
import os
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import medfilt

#AudioProcessing.py

# Storing the mp3 file
mp3_file = "HotCrossBuns.mp3"
#mp3_file = "ORIGINAL-TwinkleTwinkleLittleStar.mp3"

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

#Onset detection
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512//32) #More accurate onset times
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512//32)
print("Detected onsets at times:", onset_times)
first_onset_time = onset_times[0]# - 0.06657596 #Added the "- 0.06657596" because that was the discrepency on the 120BPM Test

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

# Read the WAV file
samplerate, data = wavfile.read(wav_file)
print(f"Estimated sample rate: {samplerate} SPS")

what_type_of_note_modifier = 4 #1 - Quarter : 2 - Eigth : 4 - 16th : 8 - 32nd
beats_per_second = tempo/60
quarter_note_duration = 1/(beats_per_second*what_type_of_note_modifier) #*2 for 8th note : *4 for 16th note : *1 for quarter note
print(f"Estimated quarter note duration: {quarter_note_duration} seconds")
#Number of samples in each frame - This typically is a power of 2 between 256-8192 (Stated in the video u posted)
frame_size = int(quarter_note_duration*samplerate) #2048 was old frame size
print(f"Estimated frame size: {frame_size} samples per frame")
#Frame 1: 0-2048, Frame 2: 1024 - 3072. Hop size is the amount of samples you skip over to start the next frame. I chose 1024 because typically each frame should contain 50% overlap. Smaller hop size, more overlap. Larger hop size, less overlap.
hop_size = int(frame_size) #1024 was old hop size

# Using the documentation to plot the WAV file
def make_plot(samplerate, data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])

    plt.plot((time*samplerate + int(first_onset_time * samplerate) % hop_size)/frame_size, data, label="Mono channel")   #FRAME = X AXIS
    #plt.plot(time, data, label="Mono channel")   #TIME  = X AXIS
    plt.axvline(x=first_onset_time, color='b', linestyle='--', alpha=0.5)
    for i in range(128):  #CHANGE HOW MANY FRAMES WE CAN SEE
        plt.axvline(x=first_onset_time+i*frame_size/samplerate, color='b', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    
#make_plot(samplerate, data)

# Compute and print the frame duration in seconds
frame_duration = hop_size / samplerate
print(f"Frame duration: {frame_duration} seconds")

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
number_of_frames = len(data) // frame_size
# Array to store the frequencies from the audio file
frequencies = []
times = []

expected_hz_values = {
        'A0': 27.50, 'A#0': 29.14, 'B0': 30.87,
        'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89, 'E1': 41.20, 'F1': 43.65, 'F#1': 46.25, 'G1': 49.00, 'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
        'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
        'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
        'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
        'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
        'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98, 'G6': 1567.98, 'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53,
        'C7': 2093.00, 'C#7': 2217.46, 'D7': 2349.32, 'D#7': 2489.02, 'E7': 2637.02, 'F7': 2793.83, 'F#7': 2959.96, 'G7': 3135.96, 'G#7': 3322.44, 'A7': 3520.00, 'A#7': 3729.31, 'B7': 3951.07,
        'C8': 4186.01
    }

#ATTEMPTING TO DO SOMETHING HUGE :D
#Seeing what big frames have an onset
frames_with_onsets = []
onset_frequencies_only = []
frames_with_onsets = np.round(onset_times * samplerate / frame_size)

#Correcting the onset frames
frame_correction_counter = 0
for i in range(len(frames_with_onsets)):
    frames_with_onsets[i] -= frame_correction_counter
    print(f"AHHHHHHHHH: {frames_with_onsets[i]}")
    if frames_with_onsets[i] % 2 == 1:
        frame_correction_counter += 1
        frames_with_onsets[i] -= 1
        print(f"CHANGING: {frames_with_onsets[i]}")

#Getting dominant frequency of those frames
for i in frames_with_onsets:
    frame_for_onsets = data[int(i*frame_size):int((i+1)*frame_size)]
    dom_freq = get_domiant_frequency(frame_for_onsets, samplerate)
    onset_frequencies_only.append(dom_freq)


# Loop through each frame and add the dominant frequency to the array
for i in range(number_of_frames):
    start = i * hop_size + int(first_onset_time * samplerate) % hop_size
    end = start + frame_size

    if end > len(data):
        frame = np.pad(data[start:], (0, end - len(data)), 'constant')
    else:
        frame = data[start:end]

    if len(frame) != frame_size:
        print(f"Skipping frame {i} due to incorrect size: {len(frame)}")
        continue
        #Note: This is skipping the last frame entirely. Will add padding to last frame to be processed at a later time

    dominant_frequency = get_domiant_frequency(frame, samplerate)
    frequencies.append(dominant_frequency)

    current_time = start / samplerate
    times.append(current_time)


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
    #    if (frame_counter % 2 == 0):
        plt.axvline(x=frame_start_time + first_onset_time, color='b', linestyle='--', alpha=0.5)
    #    else:
    #        plt.axvline(x=frame_start_time + first_onset_time, color='c', linestyle='--', alpha=0.5)
    #    frame_counter += 1
    
    
    #plt.axvline(x=first_onset_time, color='b', linestyle='--', alpha=0.5)
    #plt.axvline(x=first_onset_time+hop_size*16/sr, color='b', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_frequencies_vs_frame_index(frequencies, frame_indecies = list(range(len(frequencies)))):
    # Create a list of frame indices

    plt.figure(figsize=(12, 6))
    plt.plot(frame_indecies, frequencies, marker='o', linestyle='-', color='b')
    plt.xlabel('Frame Index')
    plt.ylabel('Dominant Frequency (Hz)')
    plt.title('Dominant Frequency vs Frame Index')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
print(f"Sample Per Second: {samplerate}")
print(f"Samples Per Frame: {frame_size}")
print(f"Seconds Per Frame: {frame_size / samplerate}")
print(f"How many frames we have: {number_of_frames}")
#print(f"Sample Rate: {samplerate}")



#uncomment this to see the graphs
#make_smoothed_dominant_frequency_graph(times, smoothed_frequencies, expected_hz_values)
#plot_frequencies_vs_frame_index(frequencies)
#plot_frequencies_vs_frame_index(onset_frequencies_only, frames_with_onsets)

print(frames_with_onsets)
print(onset_frequencies_only)

#NoteInterpretation.py -------------------------------------------------------------------------------------------------------------------------------------------------

def get_matched_notes(audio_path, dominant_frequencies_path):

    # Load audio and check if the audio data is loaded correctly
    try:
        y, sr = librosa.load(audio_path)
        if len(y) == 0:
            raise ValueError("The loaded audio signal is empty.")
        print(f"Audio loaded successfully. Signal length: {len(y)}, Sample rate: {sr}")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []
    #Load audio. Documentation: https://librosa.org/doc/latest/generated/librosa.load.html
    y, sr = librosa.load(audio_path)

    # Parameters for STE calculation 
    print(frame_size)
    print(hop_length)

    # Compute Short-Time Energy (STE)
    def compute_ste(signal, window_size, hop_length):
        if len(signal) == 0:
            raise ValueError("The input signal is empty.")
        squared_signal = np.square(signal)
        window_size = min(window_size, len(signal))
        window = np.ones(window_size) / float(window_size)
        if len(squared_signal) < window_size:
            raise ValueError("The window size is larger than the signal length.")
        ste = np.convolve(squared_signal, window, mode='same')
        return ste[::hop_length]


    # Compute STE
    ste = compute_ste(y, frame_size, hop_length)

    # Detect note onsets using STE. Documentation: https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html
    onsets = librosa.onset.onset_detect(y=y, onset_envelope=ste, sr=sr, hop_length=hop_length)
    #print(onsets)

    # No longer need to load dominant frequencies, frequencies should just exist
    print(f"Frequencies: {frequencies}")
    
    # Binary search function to find the closest frequency
    def binary_search_closest(arr, target):
        left, right = 0, len(arr) - 1
        if target <= arr[left]:
            return left
        if target >= arr[right]:
            return right
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        # Now left > right, arr[right] < target < arr[left]
        if left < len(arr) and abs(arr[left] - target) < abs(arr[right] - target):
            return left
        return right
    
    #function to interpret frequencies
    def interpret_frequencies(frequencies, expected_hz_values):
        notes = []
        # Convert the dictionary into a list of tuples
        sorted_hz_values = list(expected_hz_values.items())
        frequencies_of_notes = [item[1] for item in sorted_hz_values]
        note_names = [item[0] for item in sorted_hz_values]
        for freq in frequencies:
            note = None
            if freq <= 25.0:
                note = "Rest"
                continue
            closest_frequency_match_index = binary_search_closest(frequencies_of_notes, freq)
            error_margin = frequencies_of_notes[closest_frequency_match_index] * 0.03
            print(f"The frequency of the note matching the closest: {frequencies_of_notes[closest_frequency_match_index]}; The actual frequency: {freq}")
            if abs(freq - frequencies_of_notes[closest_frequency_match_index]) <= error_margin:
                note = note_names[closest_frequency_match_index]
            #Check if a note is a rest. Might have to lower the threshold from 30 so it doesnt confuse really low notes as rests 
            if note is None and freq < 30:
                note = "Rest"
            notes.append(note)
        return notes
        
    # Interpret frequencies into notes
    notes = interpret_frequencies(frequencies, expected_hz_values)
    print(f"Notes: {notes}")
    notes_only_onset = interpret_frequencies(onset_frequencies_only, expected_hz_values)
    print(f"Onset Only Notes: {notes_only_onset}")

    #Might no longer need this
    """ 
    # Filter out None values (unrecognized frequencies)
    filtered_notes = []
    for note in notes:
        if note in expected_hz_values.keys() or note == "Rest":
            filtered_notes.append(note)
        else:
            filtered_notes.append(None)

    print(f"Filtered Notes: {filtered_notes}")

    """
    #Removed the get_dict_notes method

    # Function to classify note durations based on STE and tempo
    def classify_note_duration(duration, beat_duration): #duration >= (4 - tolerance) * beat_duration: is standard changed to get custom
        if duration >= (4 - 0.5) * beat_duration:
            return "Whole Note"
        elif duration >= (2 - 0.3) * beat_duration:
            return "Half Note"
        elif duration >= (1 - 0.25) * beat_duration:
            return "Quarter Note"
        elif duration >= (0.5 - 0.1) * beat_duration:
            return "Eighth Note"
        elif duration >= (0.25 - 0.05) * beat_duration:
            return "Sixteenth Note"
        else:
            return "No note detected"
        
    # Segment audio based on detected onsets
    def segment_audio(y, onsets, hop_length):
        segments = []
        for i in range(len(onsets) - 1):
            start = int(onsets[i] * hop_length)
            end = int(onsets[i + 1] * hop_length)
            segment = y[start:end]
            segments.append(segment)
        # Add the last segment
        segments.append(y[int(onsets[-1] * hop_length):])
        return segments

    # Segment the audio
    segments = segment_audio(y, onsets, hop_length)

    print(f"Estimated tempo: {tempo} BPM")
    
    beat_duration = 60 / tempo

    print(tempo)
    print(f"Whole Note Min: {(4 - 0.1) * beat_duration}")
    print(f"Half Note Min: {(2 - 0.1) * beat_duration}")
    print(f"Quarter Note Min: {(1 - 0.1) * beat_duration}")
    print(f"Eigth Note Min: {(.5 - 0.1) * beat_duration}")
    print(f"Sixteenth Note Min: {(.25 - 0.1) * beat_duration}")

    # Match segments with notes based on onsets and filtered notes
    print(f"Onsets small frame indecies: {onsets}")
    print(f"Seconds of onsets {onsets * hop_length * 2 / samplerate}")
    matched_notes = []
    print(frame_duration)
    print(len(onsets))
    print(len(segments))

    
    #Remove bad segments 
    for i in range(len(segments) - 1, -1, -1):
        duration = librosa.get_duration(y=segments[i], sr=sr)
        if duration < 0.10:
            segments.pop(i)

    # Correctly match the notes with their durations
    for i, _ in enumerate(segments):
        duration = librosa.get_duration(y=segments[i], sr=sr)
        note_type = classify_note_duration(duration, beat_duration)
        note = notes_only_onset[i]
        matched_notes.append((note, duration, note_type))

    return matched_notes 

if __name__ == "__main__":
    audio_path = "HotCrossBuns.mp3"
    #audio_path = "TwinkleTwinkleLittleStar.wav"
    dominant_frequencies_path = "dominant_frequencies.npz"
    matched_notes = get_matched_notes(audio_path, dominant_frequencies_path)

    # Print matched notes with their durations and types
    for i, (note, duration, note_type) in enumerate(matched_notes):
        if note_type != "No note detected":
            print(f"Segment {i + 1}: Note: {note}, Duration: {duration:.2f} seconds, Type: {note_type}")