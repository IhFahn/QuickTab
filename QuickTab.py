from __future__ import print_function
import keras
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from tensorflow.keras import backend as K
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *
from model import *
import pickle
import sys
from scipy.io import wavfile
import librosa
from flask import Flask, request, jsonify, render_template, redirect, send_file, flash
import os
from midiutil.MidiFile import MIDIFile
from music21 import *
from tensorflow.keras.models import load_model
from read_midi import *
from simple_read_midi import *
from convert_to_midi import *
from play_music import *
import pygame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


UPLOAD_FOLDER = 'D:/Very personal/MMU/FYP/tab-cnn-master/model/upload/'



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'

@app.route('/')
def home():
    return redirect('/home.html')



@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return '', 204

@app.route('/home.html')
def home_page():
    return render_template('home.html')


@app.route('/transcribe.html')
def transcribe_page():
    return render_template('transcribe.html')


@app.route('/generation.html')
def generation_page():
    return render_template('generation.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tabcnn = TabCNN()
    tabcnn.build_model()
    tabcnn.load_weights()
    audiofile_name = os.listdir("upload")[0]
    audio_path = UPLOAD_FOLDER + audiofile_name

    preproc_mode = 'c'
    downsample = True
    normalize = True
    sr_downs = 22050

    # CQT parameters
    cqt_n_bins = 192
    cqt_bins_per_octave = 24


    # STFT parameters
    n_fft = 2048
    hop_length = 512

    con_win_size = 9
    halfwin = con_win_size // 2

    sr_original, data = wavfile.read(audio_path)
    sr_curr = sr_original
    data = data.astype(float)

    #convert input audio from stereo to mono
    #very important
    data = data.sum(axis=1) / 2

    data = librosa.util.normalize(data)
    data = librosa.resample(data, sr_original,sr_downs)
    sr_curr = sr_downs
    data = np.abs(librosa.cqt(data,
                    hop_length=hop_length,
                    sr=sr_curr,
                    n_bins=cqt_n_bins,
                    bins_per_octave=cqt_bins_per_octave))

    output = {}
    output["repr"] = np.swapaxes(data,0,1)

    np.savez('testfile', **output)
    testfile = np.load('testfile.npz')

    X_dim = (128, 192, 9, 1)
    X = np.empty(X_dim)
    full_x = np.pad(testfile["repr"], [(halfwin,halfwin), (0,0)], mode='constant')
    functionrange = -(- full_x.shape[0] // con_win_size)

    frame_idx = 0
    for i in range(functionrange-1):
        sample_x = full_x[frame_idx : frame_idx + con_win_size]
        frame_idx = frame_idx + con_win_size
        X[i,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)


    pickle.dump( X, open( "input.pkl", "wb" ) )

    inputX = pickle.load(open("input.pkl", "rb"))

    testprediction = tabcnn.model.predict(inputX)


    def stringfretextraction(testprediction):
      E = []
      A = []
      D = []
      G = []
      B = []
      e = []
      # print('E       A      D      G      B      E')
      for strings in range(testprediction.shape[0]):
        for frets in range(len(testprediction[strings][0])):
          e.append(np.argmax(testprediction[strings][5], axis = 0)-1)
          B.append(np.argmax(testprediction[strings][4], axis = 0)-1)
          G.append(np.argmax(testprediction[strings][3], axis = 0)-1)
          D.append(np.argmax(testprediction[strings][2], axis = 0)-1)
          A.append(np.argmax(testprediction[strings][1], axis = 0)-1)
          E.append(np.argmax(testprediction[strings][0], axis = 0)-1)
          break

      guitar_strings = {

        'e string': e,
        'B string': B,
        'G string': G,
        'D string': D,
        'A string': A,
        'E string': E,

      }

      df = pd.DataFrame(data=guitar_strings)
      df = df[(df['E string'] != -1) | (df['A string'] !=-1) | (df['D string'] != -1) | (df['G string'] !=-1) | (df['B string'] != -1) | (df['e string'] != -1)]

      df.loc[(df['E string'] == -1),'E string']='-'
      df.loc[(df['A string'] == -1),'A string']='-'
      df.loc[(df['D string'] == -1),'D string']='-'
      df.loc[(df['G string'] == -1),'G string']='-'
      df.loc[(df['B string'] == -1),'B string']='-'
      df.loc[(df['e string'] == -1),'e string']='-'

      df.reset_index(drop=True, inplace=True)
      df = df.T

      return df

    newguitar_df = stringfretextraction(testprediction)
    #newguitar_df.head(15)
    newguitar_df.to_pickle("./newguitar_df.pkl")
    audio_name = '/content/gdrive/My Drive/FYP/'

    os.remove(audio_path)

    return render_template('transcribe.html', tables=[newguitar_df.to_html()])#, audio = audio_path.lstrip(audio_name))

@app.route('/midi', methods = ['POST'])
def midi():

    bpm = int(request.form['bpm'])
    #bpm = int(bpm)

    tabs = pd.read_pickle("./newguitar_df.pkl")
    mf = MIDIFile(1)
    track = 0
    time = 0
    mf.addTrackName(track, time, "sample test midi")
    mf.addTempo(track, time, bpm)
    channel = 0
    volume = 100
    program = 26
    time = 0
    mf.addProgramChange(track, channel, time, program)

    for columns in range(len(tabs.columns)):
      #num1 = random.randint(1, 1)
      #num2 = random.randint(1, 1)
      num = 1

      if type(tabs[columns][0]) ==  int:
        pitch = 64 + tabs[columns][0]#midi value for open high e
        duration = num
        mf.addNote(track, channel, pitch, time, duration, volume)

      if type(tabs[columns][1]) ==  int:
        pitch = 59 + tabs[columns][1]#midi value for open b
        duration = num
        mf.addNote(track, channel, pitch, time, duration, volume)

      if type(tabs[columns][2]) ==  int:
        pitch = 55 + tabs[columns][2]#midi value for open g
        duration = num
        mf.addNote(track, channel, pitch, time, duration, volume)

      if type(tabs[columns][3]) ==  int:
        pitch = 50 + tabs[columns][3]#midi value for open d
        duration = num
        mf.addNote(track, channel, pitch, time, duration, volume)

      if type(tabs[columns][4]) ==  int:
        pitch = 45 + tabs[columns][4]#midi value for open a
        duration = num
        mf.addNote(track, channel, pitch, time, duration, volume)

      if type(tabs[columns][5]) ==  int:
        pitch = 40 + tabs[columns][5]#midi value for open low E
        duration = num
        mf.addNote(track, channel, pitch, time, duration, volume)

      time = time + num


    with open('./midi/music.mid', 'wb') as outf:
        mf.writeFile(outf)

    confirm = "The BPM that you entered is: " + str(bpm) +", you can download the file now."
    flash(confirm)
    return render_template('transcribe.html', tables=[tabs.to_html()])

@app.route('/downloadmidi', methods = ['POST'])
def downloadmidi():
    if request.method == 'POST':
        return send_file('./midi/music.mid', as_attachment=True)


@app.route('/musicgen', methods = ['POST'])
def musicgen():

    model = load_model('best_model.h5')
    audiofile_name = os.listdir("upload")[0]
    audio_path = UPLOAD_FOLDER + audiofile_name
    newmusic = simple_read_midi(audio_path)
    unique_x = pickle.load(open("unique_x.pkl", "rb"))
    frequent_notes = pickle.load(open("frequent_notes.pkl", "rb"))
    x_note_to_int = pickle.load(open("x_note_to_int.pkl", "rb"))

    tobeappend = newmusic
    newmusic = np.concatenate((newmusic, newmusic), axis=None)
    newmusic = np.concatenate((newmusic, newmusic), axis=None)
    newmusic = np.array(newmusic)
    new=[]

    for notes in newmusic:
        if notes in frequent_notes:
            new.append(notes)

    input_seq = []
    no_of_timesteps = 32
    for i in range(0, len(new) - no_of_timesteps, 1):
      newinput_ = new[i:i + no_of_timesteps]

      input_seq.append(newinput_)

    input_seq = np.array(input_seq)

    #preparing input sequences
    input_input_seq=[]
    for i in input_seq:
        temp=[]
        for j in i:
            #assigning unique integer to every note
            temp.append(x_note_to_int[j])
        input_input_seq.append(temp)

    input_input_seq = np.array(input_input_seq)

    predict = []
    for i in range(input_input_seq.shape[0]):
        """
        something = input_input_seq[i]

        new_input_input_seq = something.reshape(1,no_of_timesteps)

        prob  = model.predict(input_input_seq)[0]
        y_pred= np.argmax(prob,axis=0)
        predict.append(y_pred)

        new_input_input_seq = np.insert(input_input_seq[0],len(input_input_seq[0]),y_pred)
        new_input_input_seq = input_input_seq[1:]
        """
        something = input_input_seq[i]
        something = something.reshape(1,no_of_timesteps)
        #new_input_input_seq = np.concatenate((new_input_input_seq, something))


        prob  = model.predict(something)[0]
        y_pred= np.argmax(prob,axis=0)
        predict.append(y_pred)

        #something = np.insert(input_input_seq[0],len(input_input_seq[0]),y_pred)
        #something = something[1:]

    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
    predicted_notes = [x_int_to_note[i] for i in predict]

    outputthis = np.append(tobeappend, predicted_notes)
    pickle.dump(outputthis, open("outputthis.pkl", "wb"))

    os.remove("./upload/" + os.listdir("upload")[0])

    convert_to_midi(outputthis)
    return '', 204


@app.route('/playmidi', methods = ['POST'])
def playmidi():
    midi_filename = './midi/music.mid'
    freq = 44100  # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)

    # listen for interruptions
    try:
      # use the midi file you just saved
        play_music(midi_filename)
    except KeyboardInterrupt:
      # if user hits Ctrl/C then exit
      # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

    return '', 204













if __name__ == "__main__":
    app.run(debug=True)
