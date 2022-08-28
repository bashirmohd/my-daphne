# Convert Time Series Data to Music

The file `dataTomusic.py` converts time-series data into music and generates 2 audio files, one for the data in the `in` column and the other for the `out` column, along with the waveform for each of the audio files generated.

## Usage

The file uses python version 2.7 and multiple python libraries such as miditime, matplotlib, datetime and pygame. All of the required libraries with their versions are mentioned in `requirements.txt`

In order to install these libraries, run the following command
	pip install -r requirements.txt

The `dataTomusic.py` file takes in 4 arguments in the following order :
- input file
- name of the time column in the input file
- name of the in column in the input file
- name of the out column in the input file

### For example,
    python dataTomusic.py NERSC_SUNN.csv time in out
    python dataTomusic.py 2019_concat_data_morefeatures.csv time AMST_LOND_in AMST_LOND_out

This program creates the following files
- `data.json` which is the json version of the input csv file
- `myfileIN.mid` which is the audio for the "in" column data
- `myfileOUT.mid` which is the audio for the "out" column data
- `myfileINgraph.png` which is the audio waveform for the "in" audio file
- `myfileOUTgraph.png` which is the audio waveform for the "out" audio file

While the script is running, it prompts the user to check if any of the audios are to be played. 
If the user enters `yes`, then the audio is played. 
If the user enters `no`, then the audios are saved which can be played later using VLC or any other music software.
