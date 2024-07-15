# NotesToSheetMusic.py

import music21 as m21
from music21 import stream, duration, note, metadata
from NoteInterpretation import get_matched_notes
#Adding this to open the musicxml file
from music21 import converter, stream


# Set the MuseScore path
us = m21.environment.UserSettings()
us['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'  # Adjust this path as necessary


def create_sheet_music(matched_notes, score_name, output_path="output.musicxml"):
    # Create a Music21 Score
    score = m21.stream.Score()

    # Set the metadata for the score
    score.metadata = metadata.Metadata()
    score.metadata.title = score_name
    
    # Create a part for the score
    part = m21.stream.Part()
    score.append(part)

    # Mapping from note types to their duration in quarter lengths
    note_type_to_duration = {
        "Whole Note": 4.0,
        "Half Note": 2.0,
        "Quarter Note": 1.0,
        "Eighth Note": 0.5,
        "Sixteenth Note": 0.25
    }

    # Add matched notes to the part
    for note_info in matched_notes:
        note_name, note_duration, note_type = note_info
        if note_type != "No note detected":
            # Create a Music21 Note object
            m21_note = m21.note.Note(note_name)
            m21_note.quarterLength = note_type_to_duration[note_type]
            part.append(m21_note)

    # Write the score to a MusicXML file
    score.write('musicxml', fp=output_path)

# Example usage
#audio_path = "HotCrossBuns.mp3"
audio_path = "TwinkleTwinkleLittleStar.wav"
dominant_frequencies_path = "dominant_frequencies.npz"

# Ensure the get_matched_notes function is available and correct
matched_notes = get_matched_notes(audio_path, dominant_frequencies_path)

#create_sheet_music(matched_notes, "Hot Cross Buns")
create_sheet_music(matched_notes, "Twinkle Twinkle Little Star")

#Opening the musicxml file
file_path = "output.musicxml"
score = converter.parse(file_path)
score.show()