import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import medfilt
import os
import librosa

#NAME OF THE MP3 FILE
mp3_file = "HotCrossBuns.mp3"
#WHAT TO NAME THE WAV FILE
wav_file = "HotCrossBuns.wav"
#THE SMALLEST NOTE IN THE AUDIO
what_type_of_note_modifier = 4 #1 - Quarter : 2 - Eigth : 4 - 16th : 8 - 32nd


#Checks if the file exists
if not os.path.exists(mp3_file):
    raise FileNotFoundError(f"{mp3_file} does not exist.")

#Prints if the file is found
print("Loading MP3 file...")


#Get MP3 as audio segment
audio = AudioSegment.from_mp3(mp3_file)

# Convert the audio to mono
audio = audio.set_channels(1)

# Make the audio into a WAV file
print("Converting to WAV...")
audio.export(wav_file, format="wav")


# Load the audio file
y, sr = librosa.load(wav_file)


# Onset detection
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512//32) #More accurate onset times
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512//32)
print("Detected onsets at times:", onset_times)
first_onset_time = onset_times[0]

#Chunk to get the tempo
hop_length = 256
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
onset_env = librosa.util.normalize(onset_env)
tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
tempo = int(tempo)
print(f"Estimated tempo: {tempo} BPM")


# Read the WAV file
samplerate, data = wavfile.read(wav_file)
print(f"Estimated sample rate: {samplerate} SPS")

#Calculating framesize
beats_per_second = tempo/60
chosen_note_duration = 1/(beats_per_second*what_type_of_note_modifier)
print(f"Estimated chosen note duration: {chosen_note_duration} seconds")
frame_size = int(chosen_note_duration*samplerate) #change all hop_size to frame_sizes
print(f"Estimated frame size: {frame_size} samples per frame")

# Compute and print the frame duration in seconds
frame_duration = frame_size / samplerate
print(f"Frame duration: {frame_duration} seconds")


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



number_of_frames = len(data) // frame_size
# Array to store the frequencies and times from the audio file
frequencies = []    #MAY DELETE IF NOT ACTUALLY USED FOR ONSET DOM FREQ
times = []          #MAY DELETE IF NOT ACTUALLY USED FOR ONSET DOM FREQ


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
    




#NOTE INTERPRETATION

def get_matched_notes():

    print(f"Frequencies: {onset_frequencies_only}")

    # Table with expected Hz values for the notes
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
            closest_frequency_match_index = binary_search_closest(frequencies_of_notes, freq)
            error_margin = frequencies_of_notes[closest_frequency_match_index] * 0.03
            print(f"The frequency of the note matching the closest: {frequencies_of_notes[closest_frequency_match_index]}; The actual frequency: {freq}")
            if abs(freq - frequencies_of_notes[closest_frequency_match_index]) <= error_margin:
                note = note_names[closest_frequency_match_index]
            notes.append(note)
        return notes    
    
    """
    def interpret_note_durations():        
        note_length_by_samples = []
        for frame_ind in range(len(frames_with_onsets)):
            max_ste = 0
            number_of_frames_iterated = 0
            size_of_note_in_samples = 0
            unfound_end = True
            while unfound_end:    
                for small_frame in range(frame_size//256):
                    start = int((frames_with_onsets[frame_ind] + number_of_frames_iterated) * frame_size + small_frame * (256))
                    end = start + frame_size//256
                    frame_for_computing_ste = y[start:end]
                    ste = np.sum(np.square(frame_for_computing_ste))
                    if ste > max_ste:
                        max_ste = ste
                    else:
                        if ste < max_ste * 0.01:
                            size_of_note_in_samples = number_of_frames_iterated * frame_size + small_frame * (frame_size//256)
                            note_length_by_samples.append(size_of_note_in_samples)
                            unfound_end = False
                            break
                if frames_with_onsets[frame_ind] + number_of_frames_iterated + 1 >= number_of_frames:
                    unfound_end = False
                    size_of_note_in_samples = (number_of_frames_iterated + 1) * frame_size
                    note_length_by_samples.append(size_of_note_in_samples)
                else:
                    number_of_frames_iterated += 1
                    if frame_ind != len(frames_with_onsets) - 1:
                        if frames_with_onsets[frame_ind] + number_of_frames_iterated >= frames_with_onsets[frame_ind + 1]:
                            size_of_note_in_samples = number_of_frames_iterated * frame_size
                            note_length_by_samples.append(size_of_note_in_samples)
                            unfound_end = False
        return note_length_by_samples
    """
        
        #Go through each onset frame, convert the frame index to y[start:end] of samples, get STE, check the following frames
        #to see if the STE is below a threshold, if the STE is above then you iterate to the next frame until the frame index
        #is the same as the next loop's frame index, keep track of how many frames long a note is
    def interpret_note_durations():        
        note_length_by_frames = []
        for frame_ind in range(len(frames_with_onsets)): #Looping through each of the frames with an onset example = {27, 47, 160, 180}
            max_ste = 0
            number_of_frames_iterated = 1
            start = int((frames_with_onsets[frame_ind]) * frame_size)
            end = start + frame_size
            frame_for_computing_ste = y[start:end]
            max_ste = np.sum(np.square(frame_for_computing_ste))
            unfound_end = False
            if frame_ind == len(frames_with_onsets) - 1: #If this is the late onset, make sure to not go out of bounds
                for frame_inbetween in range(number_of_frames - int(frames_with_onsets[frame_ind])):
                    start = int((frames_with_onsets[frame_ind + frame_inbetween] + number_of_frames_iterated) * frame_size)
                    end = start + frame_size
                    frame_for_computing_ste = y[start:end]
                    ste = np.sum(np.square(frame_for_computing_ste))
                    number_of_frames_iterated = frame_inbetween + 1
                    if ste <= 0.05 * max_ste:
                        note_length_by_frames.append(number_of_frames_iterated)
                        unfound_end = False
                        break
                if unfound_end:
                    note_length_by_frames.append(number_of_frames_iterated)
            else:
                for frame_inbetween in range(int(frames_with_onsets[frame_ind + 1]) - int(frames_with_onsets[frame_ind])):
                    start = int((frames_with_onsets[frame_ind + frame_inbetween] + number_of_frames_iterated) * frame_size)
                    end = start + frame_size
                    frame_for_computing_ste = y[start:end]
                    ste = np.sum(np.square(frame_for_computing_ste))
                    number_of_frames_iterated = frame_inbetween + 1
                    if ste <= 0.05 * max_ste:
                        note_length_by_frames.append(number_of_frames_iterated)
                        unfound_end = False
                        break
                if unfound_end:
                    note_length_by_frames.append(number_of_frames_iterated)
        return note_length_by_frames     
        
        
        
        
        
        

    # Interpret frequencies into notes
    notes_only_onset = interpret_frequencies(onset_frequencies_only, expected_hz_values)
    print(f"Onset Only Notes: {notes_only_onset}")
    durations_only_onset = interpret_note_durations()
    print(f"Onset Only Durations Frames: {durations_only_onset}")
    durations_only_onset_seconds = []
    for duration in durations_only_onset:
        durations_only_onset_seconds.append(duration/samplerate*frame_size)
    print(f"Onset Only Durations Seconds: {durations_only_onset_seconds}")
    
    # Function to classify note durations based on STE and tempo
    def classify_note_duration(duration, beat_duration):
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
    
    beat_duration = 60 / tempo
    
    print(f"Whole Note Min: {(4 - 0.1) * beat_duration}")
    print(f"Half Note Min: {(2 - 0.1) * beat_duration}")
    print(f"Quarter Note Min: {(1 - 0.1) * beat_duration}")
    print(f"Eigth Note Min: {(.5 - 0.1) * beat_duration}")
    print(f"Sixteenth Note Min: {(.25 - 0.1) * beat_duration}")

    note_type = []
    for duration in durations_only_onset_seconds:
        note_type.append(classify_note_duration(duration, beat_duration))
    
    print(f"Note Types: {note_type}")

    matched_notes = []


    return matched_notes 

matched_notes = get_matched_notes()
"""
if __name__ == "__main__":
    audio_path = "HotCrossBuns.wav"
    dominant_frequencies_path = "dominant_frequencies.npz"
    matched_notes = get_matched_notes(audio_path, dominant_frequencies_path)

    # Print matched notes with their durations and types
    for i, (note, duration, note_type) in enumerate(matched_notes):
        if note_type != "No note detected":
            print(f"Segment {i + 1}: Note: {note}, Duration: {duration:.2f} seconds, Type: {note_type}")


"""