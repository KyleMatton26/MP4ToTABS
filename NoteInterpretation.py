#Note Interpretation

import numpy as np
from AudioProcessing import get_hop_size, get_samplerate
import librosa

#Load the audio path
audio_path = "HotCrossBuns.mp3"
#Documentation: https://librosa.org/doc/latest/generated/librosa.load.html
y, sr = librosa.load(audio_path)

#Documentation: https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated tempo: {tempo}")

#Load the smoothed frequencies and times 
data = np.load("dominant_frequencies.npz")
times = data["times"]
frequencies = data["frequencies"]


# Table with expected Hz values for the notes (C4, D4, E4)
expected_hz_values = {
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63
}


def interpret_frequencies(frequencies, expected_hz_values, error_margin=15):
    notes = []
    for freq in frequencies:
        note = None
        for note_name, hz_value in expected_hz_values.items():
            if abs(freq - hz_value) <= error_margin:
                note = note_name
                break
        notes.append(note)
    return notes

notes = interpret_frequencies(frequencies, expected_hz_values)

filtered_notes = []

for note in notes:
    if note is not None:
        filtered_notes.append(note)


note_durations = []
current_note = notes[0]
current_duration = 0
for note in filtered_notes:
    if note == current_note:
        current_duration += 1
    else:
        note_durations.append((current_note, current_duration))
        current_note = note
        current_duration = 1
# Append the last note and its duration
note_durations.append((current_note, current_duration))

# Convert the frame counts to time durations
note_durations_in_seconds = []
for note, duration in note_durations:
    note_durations_in_seconds.append((note, duration * get_hop_size() / get_samplerate()))


# Calculate the duration of one beat in seconds
beat_duration = 60 / tempo

# Function to classify note durations
def classify_note_duration(duration, beat_duration):
    if duration >= 4 * beat_duration:
        return "Whole Note"
    elif duration >= 2 * beat_duration:
        return "Half Note"
    elif duration >= beat_duration:
        return "Quarter Note"
    elif duration >= beat_duration / 2:
        return "Eighth Note"
    else:
        return "Sixteenth Note"

# Print the notes and their classified durations
for note, duration in note_durations_in_seconds:
    note_type = classify_note_duration(duration, beat_duration)
    print(f"Note: {note}, Duration: {duration:.2f} seconds, Type: {note_type}")



