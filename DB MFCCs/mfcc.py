import numpy, scipy, matplotlib.pyplot as plt, librosa, sklearn
import urllib.request, librosa.display, os

def convert_to_image(birdSoundPath, birdName):
    x, samp_rate = librosa.load(birdSoundPath,duration=2,sr=2*22050,offset=0.5)
    samp_rate = numpy.array(samp_rate)
    mfccs = librosa.feature.mfcc(x, sr=samp_rate, n_fft=1024, hop_length=512, n_mfcc=15)
#    mfccs = numpy.mean(librosa.feature.mfcc(y=x, sr=samp_rate, n_mfcc=13),axis=0)
    librosa.display.specshow(mfccs, sr=samp_rate, x_axis='time')
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1) 
    mfccs.var(axis=1)
    librosa.display.specshow(mfccs, sr=samp_rate, cmap="gist_yarg")
    picName = birdName[:-4] + '.png'
    save_image(picName)

def save_image(picName):
    path = 'C:\\Users\\Prashant Prakash\\Desktop\\MFCC-pics'
    fileName = path + picName
    plt.savefig(fileName, bbox_inches='tight', pad_inches=0)


def main():
   # plt.rcParams['figure.figsize'] = (14,4)

    totalFiles = 1496
    count = 1

    for file in os.listdir('C:\\Users\\Prashant Prakash\\Desktop\\DB2\\actor\\ELEN4012-Project\\DB Audio\\Unfiltered\\final_fear'):
	filename = os.fsdecode(file)

    path = 'C:\\Users\\Prashant Prakash\\Desktop\\DB2\\actor\\ELEN4012-Project\\DB Audio\\Unfiltered\\final_fear\\'
    
    
    birdSound = path + filename
    convert_to_image(birdSound, filename)
    print(filename)
    count = count + 1

if __name__ == "__main__":
    main()