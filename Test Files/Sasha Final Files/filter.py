import wave
import numpy as np
from scipy.io import wavfile
# compatibility with Python 3
#from __future__ import print_function, division, unicode_literals

# Created input file with:
# mpg123  -w 20130509talk.wav 20130509talk.mp3
wr = wave.open('ang0.wav', 'r')
par = list(wr.getparams()) # Get the parameters from the input.
# This file is stereo, 2 bytes/sample, 44.1 kHz.
par[3] = 0 # The number of samples will be set by writeframes.

# Open the output file
ww = wave.open('filtered-talk.wav', 'w')
ww.setparams(tuple(par)) # Use the same parameters as the input file.

lowpass = 2000 # Remove lower frequencies.
highpass = 3000 # Remove higher frequencies.

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
    #wavfile.write('filtsamp.wav', 44100, ns.astype(np.int16))
# Close the files.
wr.close()
ww.close()