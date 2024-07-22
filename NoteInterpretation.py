import numpy as np
import librosa
from AudioProcessing import get_frame_size, get_hop_length, get_onset_frames, get_tempo, get_samplerate, get_frame_duration
import matplotlib.pyplot as plt

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
    window_size = 512
    hop_length = get_hop_length()
    samplerate = get_samplerate()
    print(window_size)
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
    ste = compute_ste(y, window_size, hop_length)

    # Detect note onsets using STE. Documentation: https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html
    onsets = librosa.onset.onset_detect(y=y, onset_envelope=ste, sr=sr, hop_length=hop_length)
    #print(onsets)

    #Not using STE for testing
    #onsets = get_onset_frames()

    # Load dominant frequencies
    data = np.load(dominant_frequencies_path)
    frequencies = data["frequencies"]
    print(f"Frequencies: {frequencies}")

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

    # Filter out None values (unrecognized frequencies)
    filtered_notes = []
    for note in notes:
        if note in expected_hz_values.keys() or note == "Rest":
            filtered_notes.append(note)
        else:
            filtered_notes.append(None)

    print(f"Filtered Notes: {filtered_notes}")

    

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

    tempo = get_tempo()
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
    frame_duration = get_frame_duration()
    print(frame_duration)
    print(len(onsets))
    note_index = 0
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

        #Current problem: the +1 to the index shift. sometimes the index shift should have a +1 and sometimes it shouldnt. for example when a half note is 5 frames long it needs a +1 and when a half note is 4 frames long it doesnt need a +1
        #Update to problem: For twinkle twinkle, there are 3 frames together for the first 2 quarter notes. This approach does not take that into account. Now we would need a condition to add a -1. Might be better to find a different solution
        #Also, I chose to use duration instead of note_type because note_type could be inconsistent because sometimes, for example, a half note is 5 frames when other half notes are four frames.
        if note_index < len(filtered_notes):
            note = filtered_notes[note_index]
            matched_notes.append((note, duration, note_type))
            print("Duration: " + str(duration))
            i = note_index + 1
            count = 1
            while note == filtered_notes[note_index] and i < len(filtered_notes):
                if filtered_notes[i] == note:
                    count += 1
                    note = filtered_notes[i]
                    i += 1
                else:
                    break
                    
            if count % 2 == 1:
                index_shift = (round(duration / frame_duration))# + 1
            else:
                index_shift = (round(duration / frame_duration))

            print("Index Shift: " + str(index_shift))
            note_index += index_shift
            print("Note Index: " + str(note_index))

    return matched_notes 

if __name__ == "__main__":
    audio_path = "HotCrossBuns.wav"
    #audio_path = "TwinkleTwinkleLittleStar.wav"
    dominant_frequencies_path = "dominant_frequencies.npz"
    matched_notes = get_matched_notes(audio_path, dominant_frequencies_path)

    # Print matched notes with their durations and types
    for i, (note, duration, note_type) in enumerate(matched_notes):
        if note_type != "No note detected":
            print(f"Segment {i + 1}: Note: {note}, Duration: {duration:.2f} seconds, Type: {note_type}")
