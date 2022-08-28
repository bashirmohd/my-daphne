import csv
import sys
import json
import midi
import pprint
import datetime
import pygame, base64
import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

from miditime.miditime import MIDITime


jsonFilePath = 'data.json'

''' Converts the inputfile from csv format to json format '''
def convertCSVToJSON(inputfile, time, inCol, outCol):
    # Output file
    

    # Read csv file
    print("Converting .csv to .json")
    data = []
    with open(inputfile) as csvFile:
        # Creating a csv file reader
        csvReader = csv.DictReader(csvFile)
        # Writing data line by line
        for csvRow in csvReader:
            d = {} # Creating a dictionary
            d['time'] = csvRow[time]
            d['in'] = float(csvRow[inCol])
            d['out'] = float(csvRow[outCol])
            data.append(d)

    # Writing the data in json format using dumps() function
    with open(jsonFilePath, 'w') as jsonFile:
        jsonFile.write(json.dumps(data, indent=4))

''' Find the min value in the 'in' data column '''
def in_min_data_value(dataArray):
    #data is a dictionary with index and different columns like time, in and out. In order to access the value at an index in a column we write data[i]['time']
    #setting dummy minimum
    in_min_val = dataArray[0]['in']
    i = 0
    #iterating through the array to find the minimum value
    while(i < len(dataArray)):
        if((in_min_val) > (dataArray[i]['in'])): 
            in_min_val = dataArray[i]['in']
        i = i + 1
    return in_min_val

'''find min value in 'out' data column '''
def out_min_data_value(dataArray):
    #setting dummy minimum
    out_min_val = dataArray[0]['out']
    i = 0
    #iterating through the array to find the minimum value
    while(i < len(dataArray)):
        if(out_min_val > dataArray[i]['out']):
            out_min_val = dataArray[i]['out']
        i = i + 1
    return out_min_val

#Step 6
'''find max value in 'in' data column '''
def in_max_data_value(dataArray):
    #setting dummy maximum
    in_max_val = dataArray[0]['in']
    i = 0
    #iterating through the array to find the maximum value
    while(i < len(dataArray)):
        if(in_max_val) < (dataArray[i]['in']):
            in_max_val = dataArray[i]['in']
        i = i + 1
    return in_max_val

'''find max value in 'out' data column '''
def out_max_data_value(dataArray):
    #setting dummy maximum
    out_max_val = dataArray[0]['out']
    i = 0
    #iterating through the array to find the maximum value
    while(i < len(dataArray)):
        if(out_max_val < dataArray[i]['out']):
            out_max_val = dataArray[i]['out']
        i = i + 1
    return out_max_val

#Step 7
'''function find the pitches of the data points at each row in 'in' column by taking in the maximum and minimum value of the column and a major to play in'''
def in_mag_to_pitch_tuned(minValue, maxValue, input, RangeOfNote):
    # Find where does the data point sit in the domain of the data
    in_scale_pct = mymidiIN.linear_scale_pct(minValue, maxValue, input)

    # Another option: Linear scale, reverse order
    #scale_pct = mymidi.linear_scale_pct(minValue, maxValue, input, true)
    # Another option: Logarithmic scale, reverse order (can be true or false)
    #scale_pct = mymidi.log_scale_pct(minValue, maxValue, in, false)

    # Find the note in the given major that matches your data point 
    in_note = mymidiIN.scale_to_note(in_scale_pct, RangeOfNote)

    #Translate that note to a MIDI pitch
    in_midi_pitch = mymidiIN.note_to_midi_pitch(in_note)

    return in_midi_pitch

'''function find the pitches of the data points at each row in 'in' column by taking in the maximum and minimum value of the column and a major to play in'''
def out_mag_to_pitch_tuned(minValue, maxValue, input, RangeOfNote):
    # find where does the data point sit in the domain of the data
    out_scale_pct = mymidiOUT.linear_scale_pct(minValue, maxValue, input)

    # Another option: Linear scale, reverse order
    #scale_pct = mymidi.linear_scale_pct(minValue, maxValue, input, true)
    # Another option: Logarithmic scale, reverse order (can be true or false)
    #scale_pct = mymidi.log_scale_pct(minValue, maxValue, in, false)

    #Find the note in the given major that matches your data point 
    out_note = mymidiOUT.scale_to_note(out_scale_pct, RangeOfNote)

    #Translate that note to a MIDI pitch
    out_midi_pitch = mymidiOUT.note_to_midi_pitch(out_note)

    return out_midi_pitch


def plotGraph(data_array_1, data_array_2):
    
    song = midi.read_midifile('myfileIN.mid')
    song.make_ticks_abs()
    tracks = []
    i = 0
    for track in song:
        notes = [note for note in track if note.name == 'Note On']
        pitch = [note.pitch for note in notes]
        tick = [note.tick for note in notes]
        tracks += [data_array_1, pitch]

    #plt.plot(*tracks)
    #plt.ylabel("Pitch")
    #plt.xlabel("Minutes")
    #plt.show()
    #plt.savefig('myfileINgraph.png')

    #plt.clf()

    song = midi.read_midifile('myfileOUT.mid')
    #print(song)
    song.make_ticks_abs()
    tracks = []
    i = 0 
    for track in song:
        notes = [note for note in track if note.name == 'Note On']
        #print([note for note in track])
        pitch = [note.pitch for note in notes]
        tick = [note.tick for note in notes]
        tracks += [data_array_2, pitch]

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint("printing tracks array")
    # pp.pprint(tracks)

    #plt.plot(*tracks)
    #plt.ylabel("Pitch")
    #plt.xlabel("Minutes")
    # plt.show()
    #plt.savefig('myfileOUTgraph.png')

#Step 10 - playing the file when the script is run
def midi_to_base64(midi_filename):
  return base64.encodestring(open(midi_filename, 'rb').read())

def play_music(midi_filename):
  '''Stream music_file in a blocking manner'''
  clock = pygame.time.Clock()
  pygame.mixer.music.load(midi_filename)
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy():
    clock.tick(30) # check if playback has finished

''' main method '''
if __name__ == "__main__":

    # Instantiate the midi class with a tempo (120bpm is the default) and an output file destination.
    mymidiIN = MIDITime(120 ,'myfileIN.mid', 12, 5, 1)  
    mymidiOUT = MIDITime(120, 'myfileOUT.mid', 12, 5, 1)

    #converting the input file to json format using convertCSVToJSON()
    if len(sys.argv) > 1:
            convertCSVToJSON(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    #creating 2 arrays for data in the 'in' and 'out' columns where the time is a datetime object
    my_data_in_timed = []
    my_data_out_timed = []

     #creating 2 arrays for data in the 'in' and 'out' columns where the datetime object is converted to a beat using beat() in miditime library
    my_in_data_epoched_hours = []
    my_out_data_epoched_hours = []

    #Step 2
    #Read from the json file
    #jsonFilePath = 'data.json'
    with open(jsonFilePath) as jsonFile:
        data = json.load(jsonFile)
        i = 0
        #reading line by line
        while( i < len(data)):

             #Step 3
            #getting a datetime object from the time column in the database
           
            date_string = data[i][sys.argv[2]]
            date_time_obj = None
            try:
                format = "%Y-%m-%d %H:%M:%S"
                datetime.datetime.strptime(date_string, format)
                date_time_obj = datetime.datetime.strptime(data[i][sys.argv[2]], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    format = "%m/%d/%y %H:%M"
                    print(format, date_string)
                    datetime.datetime.strptime(date_string, format)
                    # print("true")
                    date_time_obj = datetime.datetime.strptime(data[i]['time'], '%m/%d/%y %H:%M')
                except ValueError:
                    print("")
                
            #array with columns ['hours_since_epoch', 'in'].
            #in each row 'hours_since_epoch' has the time in terms of hours elapsed since epoch along with the corresponding value given in the 'in' column
            #hours since epoch is taken out using days_since_epoch() * 24
            my_in_data_epoched_hours.append(
                {'hours_since_epoch': mymidiIN.days_since_epoch(date_time_obj) * 24, 'in': data[i]['in']}
            )

            #array with columns ['hours_since_epoch', 'out'].
            #in each row 'hours_since_epoch' has the time in terms of hours elapsed since epoch along with the corresponding value given in the 'out' column
            #hours since epoch is taken out using days_since_epoch() * 24
            my_out_data_epoched_hours.append(
                {'hours_since_epoch': mymidiOUT.days_since_epoch(date_time_obj) * 24, 'out': data[i]['out']}
            )

            #Step 4
            #array with columns ['beat', 'in'].
            #in each row 'beat' is the time at which a particular note should occur in the song along with the corresponding 'in' value 
            #calculate beat for each row using the beat() function from the corresponding hours_since_epoch values
            my_data_in_timed.append(
                {'beat': mymidiIN.beat(my_in_data_epoched_hours[i]['hours_since_epoch'])*(365.25/24),'in': my_in_data_epoched_hours[i]['in']}
            )
            
            #array with columns ['beat', 'out'].
            #in each row 'beat' is the time at which a particular note should occur in the song along with the corresponding 'out' value 
            #calculate beat for each row using the beat() function from the corresponding hours_since_epoch values
            my_data_out_timed.append(
                {'beat': mymidiOUT.beat(my_out_data_epoched_hours[i]['hours_since_epoch'])*(365.25/24), 'out': my_out_data_epoched_hours[i]['out']}
            )

            i = i + 1

    #executing step 5
    #calculating min value in 'in' column using in_min_data_value()
    inMinVal = in_min_data_value(my_data_in_timed)
    #calculating min value in 'out' column using out_min_data_value()
    outMinVal = out_min_data_value(data)

    #executing step 6
    #calculating max value in 'in' column using in_max_data_value()
    inMaxVal = in_max_data_value(my_data_in_timed)
    #calculating max value in 'out' column using out_max_data_value() 
    outMaxVal = out_max_data_value(my_data_out_timed)

    #excuting step 7
    #creating a list that stores the time at which a note is played, note to be played, attack and duration of each data point in the 'in' column
    in_note_list = []
    #creating a list that stores the time at which a note is played, note to be played, attack and duration of each data point in the 'in' column
    out_note_list = []

    #taking out the time the first datapoint in the column is actually played considering the first note to be played is from 1st Jan 1970
    start_in_time = my_data_in_timed[0]['beat']
    start_out_time = my_data_out_timed[0]['beat']

    #Declaring notes for different majors
    a_major = ['A', 'B', 'C#','D', 'E', 'F#', 'G#', 'A']
    b_major = ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#', 'B']
    c_major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    d_major = ['D','E', 'F#','G','A','B', 'C#','D']
    e_major = ['E', 'F#','G#','A','B','C#','D#','E']
    f_major = ['F', 'G', 'A', 'Bb', 'C', 'D', 'E', 'F']
    g_major = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G']

    #iterating over the data to fill in the above arrays 
    i = 0
    while(i < len(data)):
        in_note_list.append([
            my_data_in_timed[i]['beat'] - start_in_time, # time of each data point relative to the time of the first datapoint in each database
            in_mag_to_pitch_tuned(inMinVal, inMaxVal, my_data_in_timed[i]['in'], e_major), #note
            80,  # attack
            1  # duration of notes, in beats
        ])
            
        out_note_list.append([
            #relative hours since 1st june 2018
            my_data_out_timed[i]['beat'] - start_out_time, # time of each data point relative to the time of the first datapoint in each database
            out_mag_to_pitch_tuned(outMinVal, outMaxVal, my_data_out_timed[i]['out'], e_major), #note
            100,  # attack
            1  # duration of notes, in beats
        ])
    
        i = i + 1


    #Step 8
    # Add a track with those notes using MIDITime's add_track() method
    mymidiIN.add_track(in_note_list)
    mymidiOUT.add_track(out_note_list)

    #Step 9
    # Saving the .mid file using MIDITime's save_midi() method
    print("2")
    mymidiIN.save_midi()
    mymidiOUT.save_midi()
    print("saved both created audios")

    #Step 10
    #using pythons PyGame library to play the audios when the script is run 
    # mixer config 
    freq = 44100  # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)

    #Printing graphs
    OutDataSeconds = []
    for i in range(0, len(data)):
        OutDataSeconds.append((my_data_out_timed[i]['beat'] - start_out_time)/2/60)

    InDataSeconds = []
    for i in range(0, len(data)):
        InDataSeconds.append((my_data_in_timed[i]['beat'] - start_in_time)/2/60)

    #Step 11
    #plotting the graph for the created audios and saving them 
    plotGraph(InDataSeconds, OutDataSeconds)
    print("saved the graphs for both audios")
    print("size of graph files ", len(InDataSeconds), len(OutDataSeconds))
    print("size of input files ", len(jsonFilePath))

    # listen for interruptions
    try:
        # use the midi file you just saved to play it 
        inputStr = input("do you want to hear the audio file for 'in' column (YES/NO)?")
        if(inputStr.upper() == 'YES'):
            print("printing in audio file")
            play_music('myfileIN.mid')

        # use the midi file you just saved to play it 
        inputStr = input("do you want to hear the audio file for 'out' column (YES/NO)?")
        if(inputStr.upper() == 'YES'):
            print("printing out audio file")
            play_music('myfileOUT.mid')
    except KeyboardInterrupt:
    # if user hits Ctrl/C then exit
    # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

   