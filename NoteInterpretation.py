import numpy as np
import librosa
from AudioProcessing import get_frame_size, get_hop_size

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
    window_size = get_frame_size()
    hop_length = get_hop_size()

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
    onsets = librosa.onset.onset_detect(onset_envelope=ste, sr=sr, hop_length=hop_length, backtrack=False)

    # Load dominant frequencies
    data = np.load(dominant_frequencies_path)
    frequencies = data["frequencies"]

    # Table with expected Hz values for the notes
    expected_hz_values = {
        #HCB expected values
        #'C4': 261.63,
        #'D4': 293.66,
        #'E4': 329.63
        
        
        #TTLS expected values
        'C4': 261.63,
        'D4': 293.66,
        'E4': 329.63,
        'G4': 392.00,
        'A4': 440.00,
        'F4': 349.23
    }

    # Function to interpret frequencies into notes
    def interpret_frequencies(frequencies, expected_hz_values):
        notes = []
        for freq in frequencies:
            note = None
            for note_name, hz_value in expected_hz_values.items():
                #trying to make the error margin better
                error_margin = hz_value * 0.03
                if abs(freq - hz_value) <= error_margin:
                    note = note_name
                    break
            notes.append(note)
        return notes

    # Interpret frequencies into notes
    notes = interpret_frequencies(frequencies, expected_hz_values)

    # Filter out None values (unrecognized frequencies)
    filtered_notes = []
    for note in notes:
        if note in expected_hz_values.keys():
            filtered_notes.append(note)
        else:
            filtered_notes.append(None)

    # Function to classify note durations based on STE and tempo
    def classify_note_duration(duration, beat_duration, tolerance=0.1):
        if duration >= (4 - tolerance) * beat_duration:
            return "Whole Note"
        elif duration >= (2 - tolerance) * beat_duration:
            return "Half Note"
        elif duration >= (1 - tolerance) * beat_duration:
            return "Quarter Note"
        elif duration >= (0.5 - tolerance) * beat_duration:
            return "Eighth Note"
        elif duration >= (0.25 - tolerance) * beat_duration:
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

    # Calculate tempo. Documentation: https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    beat_duration = 60 / tempo

    # Match segments with notes based on onsets and filtered notes
    matched_notes = []
    for i, onset in enumerate(onsets):
        duration = librosa.get_duration(y=segments[i], sr=sr)
        note_type = classify_note_duration(duration, beat_duration)

        # Ensure onset is within range of filtered_notes and skip if no note is detected
        if onset < len(filtered_notes) and filtered_notes[onset]:
            note = filtered_notes[onset]
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
