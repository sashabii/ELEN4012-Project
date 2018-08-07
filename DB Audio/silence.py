import librosa
import os
import numpy as np
from pydub import AudioSegment

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
   
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

for folder in os.listdir("Original Files"):
    foldername = os.fsdecode(folder)
    print(foldername)
    for file in os.listdir('Original Files\\%s' % foldername):
        filename = os.fsdecode(file)
        sound = AudioSegment.from_file('Original Files\\%s\\%s' % (foldername,filename), format="wav")
       
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)    
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export("Removed Silence\\%s" % filename, format="wav") #Exports to a wav file in the current path.
