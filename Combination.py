import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq
import os
import librosa

#NAME OF THE MP3 FILE
mp3_file = "RestTest.mp3"





wav_file = mp3_file.replace(".mp3", ".wav")

#Check if the file exists
if not os.path.exists(mp3_file):
    raise FileNotFoundError(f"{mp3_file} does not exist.")

#Convert MP3 to WAV
print("Loading MP3 file...")
audio = AudioSegment.from_mp3(mp3_file)
audio = audio.set_channels(1)  #Convert to mono
print("Converting to WAV...")
audio.export(wav_file, format="wav")

#Load WAV file
y, sr = librosa.load(wav_file)
print(f"Estimated sample rate: {sr} SPS")

#Onset detection
hop_length = 256
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length//8)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length//8)
print("Detected onsets at times:", onset_times)

#Tempo detection
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
tempo = int(tempo)
print(f"Estimated tempo: {tempo} BPM")

#Calculate samples per beat
beats_per_second = tempo / 60
seconds_per_beat = 1 / beats_per_second
samples_per_beat = seconds_per_beat * sr
print(f"Samples per beat: {samples_per_beat}")

#Utility functions
def get_dominant_frequency(frame, samplerate):
    fft_signal = rfft(frame)
    fft_signal_abs = np.abs(fft_signal)
    frequencies = rfftfreq(len(frame), 1 / samplerate)
    dominant_frequency_index = np.argmax(fft_signal_abs)
    dominant_frequency = frequencies[dominant_frequency_index]
    return dominant_frequency

def compute_short_time_energy(frame):
    return np.sum(frame ** 2)

#Getting dominant frequency and note durations
onset_frequencies_only = []
note_durations = []
getting_dom_freq_window_size = int(samples_per_beat // 4)

for i, onset in enumerate(onset_times):
    starting_sample = int(onset * sr)
    ending_sample = starting_sample + getting_dom_freq_window_size
    if ending_sample > len(y):  #Ensure we don't go out of bounds
        ending_sample = len(y)
    frame_for_onsets = y[starting_sample:ending_sample]
    dom_freq = get_dominant_frequency(frame_for_onsets, sr)
    onset_frequencies_only.append(dom_freq)

    #Loop through frames until next onset or STE below threshold
    current_sample = starting_sample
    next_onset_sample = int(onset_times[i + 1] * sr) if i + 1 < len(onset_times) else len(y)
    previous_STE = 0
    hasRest = False
    while current_sample < next_onset_sample:
        frame = y[current_sample:current_sample + getting_dom_freq_window_size]
        if len(frame) < getting_dom_freq_window_size:
            break  #Break if the frame is too short (end of data)
        ste = compute_short_time_energy(frame)
        if ste < previous_STE * 0.33: #THE 0.33 IS THE DROP FROM THE PREVIOUS NOTE
            hasRest = True
            break  #Break if the STE is below the threshold (note end)
        current_sample += getting_dom_freq_window_size
        previous_STE = ste

    #Calculate note duration
    note_duration_samples = current_sample - starting_sample
    note_duration_seconds = note_duration_samples / sr
    note_durations.append(note_duration_seconds)
    if hasRest:
        note_durations.append((current_sample - next_onset_sample) / sr)
#If the last note is a "rest" by being negative it will just remove the ending rest
if note_durations and note_durations[-1] < 0:
    note_durations.pop()

#NOTE INTERPRETATION
print(f"Frequencies: {onset_frequencies_only}")
print(f"Note Durations (seconds): {note_durations}")

#Table with expected Hz values for the notes
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

#Binary search function to find the closest frequency
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
    if left < len(arr) and abs(arr[left] - target) < abs(arr[right] - target):
        return left
    return right

#Function to interpret frequencies
def interpret_frequencies(frequencies, expected_hz_values):
    notes = []
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

#Interpret frequencies into notes
notes_only_onset = interpret_frequencies(onset_frequencies_only, expected_hz_values)
print(f"Onset Only Notes: {notes_only_onset}")

#Function to convert durations to note types
def duration_to_note_type(duration_seconds, tempo):
    beats_per_second = tempo / 60
    beat_duration = 1 / beats_per_second  #Duration of one beat in seconds
    note_durations_in_beats = duration_seconds / beat_duration
    if duration_seconds < 0:
        if note_durations_in_beats <= -3.5:
            return "Whole Rest"
        elif note_durations_in_beats <= -1.75:
            return "Half Rest"
        elif note_durations_in_beats <= -0.875:
            return "Quarter Rest"
        elif note_durations_in_beats <= -0.4375:
            return "Eighth Rest"
        elif note_durations_in_beats <= -0.21875:
            return "Sixteenth Rest"

    if note_durations_in_beats >= 3.5:
        return "Whole"
    elif note_durations_in_beats >= 1.75:
        return "Half"
    elif note_durations_in_beats >= 0.875:
        return "Quarter"
    elif note_durations_in_beats >= 0.4375:
        return "Eighth"
    elif note_durations_in_beats >= 0.21875:
        return "Sixteenth"
    else:
        return "Unknown Duration"

#Convert note durations to note types
note_types = [duration_to_note_type(duration, tempo) for duration in note_durations]

notes = []
i = 0
for note in note_types:
    if "Rest" not in note:
        notes.append((notes_only_onset[i], note))
        i += 1
    else:
        notes.append(("Rest", note))

def get_notes():
    return notes


#Print the notes and their corresponding types
print(f"Onset Only Notes: {notes_only_onset}")
print(f"Note Durations (seconds): {note_durations}")
print(f"Note Types: {note_types}")
print(notes)