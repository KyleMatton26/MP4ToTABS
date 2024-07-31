import music21 as m21
from music21 import stream, duration, note, metadata
from Combination import get_notes
from music21 import converter

# Set the MuseScore path
us = m21.environment.UserSettings()
us['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'  # Adjust this path as necessary

def create_sheet_music(notes, score_name, output_path="output.musicxml"):
    # Create a Music21 Score
    score = m21.stream.Score()

    # Set the metadata for the score
    score.metadata = metadata.Metadata()
    score.metadata.title = score_name
    
    # Create a part for the score
    part = m21.stream.Part()
    score.append(part)

    # Mapping from note durations to their quarter lengths
    note_type_to_duration = {
        "Whole": 4.0,
        "Half": 2.0,
        "Quarter": 1.0,
        "Eighth": 0.5,
        "Sixteenth": 0.25,
        "Whole Rest": 4.0,
        "Half Rest": 2.0,
        "Quarter Rest": 1.0,
        "Eighth Rest": 0.5,
        "Sixteenth Rest": 0.25
    }

    # Add notes to the part
    for note_info in notes:
        note_name, note_duration = note_info
        if "Rest" in note_duration:
            # Create a Music21 Rest object
            m21_rest = m21.note.Rest()
            m21_rest.quarterLength = note_type_to_duration[note_duration]
            part.append(m21_rest)
        else:
            # Create a Music21 Note object
            m21_note = m21.note.Note(note_name)
            m21_note.quarterLength = note_type_to_duration[note_duration]
            part.append(m21_note)

    # Write the score to a MusicXML file
    score.write('musicxml', fp=output_path)

# Example usage
audio_path = "RestTest.wav"

# Ensure the get_notes function is available and correct
notes = get_notes()
#Make a way to make score name correct
#score_name = ____
create_sheet_music(notes, "Rest Test")

# Opening the musicxml file
file_path = "output.musicxml"
score = converter.parse(file_path)
score.show()
