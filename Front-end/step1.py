import pyaudio
import wave
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import os
import matplotlib.pyplot as plt
import numpy, scipy, matplotlib.pyplot as plt, librosa, sklearn
import urllib.request, librosa.display, os


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()



# FILTERING




wr = wave.open('C:\\Users\\Prashant Prakash\\Desktop\\voice.wav','r')
par=list(wr.getparams())
# This file is stereo, 2 bytes/sample, 44.1 kHz.
par[3] = 0 # The number of samples will be set by writeframes.
# Open the output file
ww =  wave.open('C:\\Users\\Prashant Prakash\\Desktop\\filtered\\filtvoice.wav', 'w')
ww.setparams(tuple(par)) # Use the same parameters as the input file.
lowpass = 3825 # Remove lower frequencies.
highpass = 7000 # Remove higher frequencies.

sz = wr.getframerate() # Read and process 1 second at a time.
c = int(wr.getnframes()/sz) # whole file
for num in range(c):
	print('Processing {}/{} s'.format(num+1, c))
	da = np.fromstring(wr.readframes(sz), dtype=np.int16)
	left, right = da[0::2], da[1::2] # left and right channel
	lf, rf = np.fft.rfft(left), np.fft.rfft(right)
	lf[:lowpass], rf[:lowpass] = 0, 0 # low pass filter
	lf[55:66], rf[55:66] = 0, 0 # line noise
	lf[highpass:], rf[highpass:] = 0,0 # high pass filter
	nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
	ns = np.column_stack((nl,nr)).ravel().astype(np.int16)
	ww.writeframes(ns.tobytes())
# Close the files.
wr.close()
ww.close()


#MFCC
for file in os.listdir('C:\\Users\\Prashant Prakash\\Desktop\\filtered'):
    filename = os.fsdecode(file)
   
    path = 'C:\\Users\\Prashant Prakash\\Desktop\\filtered\\'
    
    
    birdSound = path + filename
    #convert_to_image(birdSound, filename)
    print(filename)


    x, samp_rate = librosa.load(birdSound,duration=2,sr=2*22050,offset=0.5)
    samp_rate = numpy.array(samp_rate)
    mfccs = librosa.feature.mfcc(x, sr=samp_rate, n_fft=1024, hop_length=512, n_mfcc=15)
#   mfccs = numpy.mean(librosa.feature.mfcc(y=x, sr=samp_rate, n_mfcc=13),axis=0)
    librosa.display.specshow(mfccs, sr=samp_rate, x_axis='time')
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1) 
    mfccs.var(axis=1)
    librosa.display.specshow(mfccs, sr=samp_rate, cmap="gist_yarg")
    picName = filename[:-4] + '.png'
    #save_image(picName)

    path1 = 'C:\\Users\\Prashant Prakash\\Desktop\\'
    fileName = path1 + picName
    plt.savefig(fileName, bbox_inches='tight', pad_inches=0)


