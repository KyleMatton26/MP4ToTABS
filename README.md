# MP3ToSheetMusic

Convert MP3 audio files into sheet music using Python. This project analyzes MP3 files to detect musical notes and their durations, then generates a MusicXML file for viewing in music notation software.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1. **Python**: [Download Python](https://www.python.org/)
2. **MuseScore**: [Download MuseScore](https://musescore.org/)
3. **FFmpeg**: [Install FFmpeg](https://ffmpeg.org/download.html)

## 🚀 Installation

Follow these steps to get your development environment set up:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git

2. **Navigate to the Project Directory**
   ```bash
   cd your-repo-name

3. **Install Dependencies**
   ```bash
   pip install numpy pydub scipy librosa music21

4. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git


## 🎵 Usage

To convert your MP3 file to sheet music:

1. **Prepare Your MP3 File**
   Place your MP3 file in the project directory (e.g., `HotCrossBuns.mp3`).

2. **Run the Audio Analysis Script**
   ```bash
   python NoteInterpretation.py


   This script will:
   - Convert the MP3 file to WAV format
   - Analyze the audio for onsets, tempo, and frequencies
   - Calculate note durations and interpret notes

3. **Run the MusicXML File Creation Script**
   ```bash
   python NotesToSheetMusic.py


   This script will:
   - Convert the arrays into a sheet music
   - Will display the sheet music in MuseScore

## 🔧 Customization
To customize the code to read the proper files:
1. MP3 File Name: Update the mp3_file variable in NoteInterpretation.py to match your MP3 file.
2. MuseScore Path: Adjust the MuseScore path in CreateSheetMusic.py if MuseScore is installed in a different location.
















---------------------------------------------------------------------------------------------------------------------------------------
This project will convert an MP4 file, eventually spotify songs, to TABS or sheet music.

Instructions on how to get this running on MacOS - Type all commands without the quotation marks enclosing them

1) Go to https://github.com/KyleMatton26/MP4ToTABS
2) Navigate in the terminal to where you want to save the repository using commands like "ls", "cd nameOfPlace", and "cd .."
2) Type "git clone https://github.com/KyleMatton26/MP4ToTABS.git"
3) Open VS Code and at the top click "File" -> "Open Folder...", then navigate to the "MP4ToTabs" folder that was cloned and click open
4) Open the terminal in VS Code by going to the top of the screen and clicking "Terminal" then "New Terminal"
5) Install HomeBrew by pasting "/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"" into the terminal and pressing "return" or "enter"
6) Type "pip install ______" replacing the "______" with a library, make sure to do that for all the libraries used
7) Type "brew install ______" replacing the "______" with things like "ffmpeg" UNSURE IF THIS WORKS, MIGHT NOT NEED HOMEBREW EVEN!!!!!!!
8) Click run on the following files in order: "AudioProcessing.py", "NoteInterpretation.py", and then "NotesToSheetMusic.py"
9) Follow these instructions to modify https://github.com/firstcontributions/first-contributions/blob/main/README.md


*** Don't think I actually needed HomeBrew since I don't believe any installs worked. However, it didn't work the first time and then I did run the "brew install ffmpeg" and it worked, so maybe something did install and change ***