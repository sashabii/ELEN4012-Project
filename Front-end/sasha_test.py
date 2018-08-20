import sys, pyaudio, wave,urllib.request, librosa.display, os, pickle, cv2
import numpy as np
from PyQt4 import QtGui, QtCore
import numpy, scipy, matplotlib.pyplot as plt, librosa, sklearn
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Recording Variables
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "tmp\\recording.wav"
FILTERED_OUTPUT_FILENAME= "tmp\\filtered.wav"
MFCC_OUTPUT_FILENAME = "tmp\\mfcc.png"
MODEL_FILE = "model\\model.model"
PICKLE_FILE = "model\\lb.pickle"

# Filter Variables
lowpass = 1300 
highpass = 6000 

class Window(QtGui.QMainWindow):

	# Core Application
	# Template for GUI - constant
	def __init__(self):
		super(Window,self).__init__()
		self.setGeometry(50,50,500,300)
		self.setWindowTitle("Emotion Detector")
		self.home()

	def home(self):
		# ~~~~~ TEXT ~~~~~
		title = QtGui.QLabel("Emo Detect",self)
		title.setText("Emotion Detector")
		title.move(200,0)

		self.status = QtGui.QLabel("",self)
		self.status.move(200,100)

		self.classification = QtGui.QLabel("",self)
		self.classification.move(200,150)

		# ~~~~~ BTNS ~~~~~
		recBtn = QtGui.QPushButton("REC", self)
		recBtn.clicked.connect(self.recordVoice)
		recBtn.move(200,50)
		
		quitBtn = QtGui.QPushButton("QUIT", self)
		quitBtn.clicked.connect(self.closeApp)
		quitBtn.move(200,200)
		
		# ~~~~~ IMGS ~~~~~
		self.show()


	def recordVoice(self):
		self.status.setText("Recording...")
		self.status.repaint()
		p = pyaudio.PyAudio()
		stream = p.open(format=FORMAT, channels=CHANNELS,rate=RATE,
                        input=True,frames_per_buffer=CHUNK)
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

		self.status.setText("Recording Complete!")
		self.status.repaint()

		self.filterVoice()
        
		
	def filterVoice(self):
		self.status.setText("Filtering...")
		self.status.repaint()

		wr = wave.open(WAVE_OUTPUT_FILENAME,'r')
		par=list(wr.getparams())
		# This file is stereo, 2 bytes/sample, 44.1 kHz.
		par[3] = 0 # The number of samples will be set by writeframes.
		# Open the output file
		ww =  wave.open(FILTERED_OUTPUT_FILENAME, 'w')
		ww.setparams(tuple(par)) # Use the same parameters as the input file.

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
		self.status.setText("Filtering Complete!")
		self.status.repaint()

		self.convertMFCC()

	def convertMFCC(self):
		self.status.setText("Converting to MFCC...")
		self.status.repaint()

		x, samp_rate = librosa.load(FILTERED_OUTPUT_FILENAME,duration=2,sr=2*22050,offset=0.5)
		samp_rate = numpy.array(samp_rate)
		mfccs = librosa.feature.mfcc(x, sr=samp_rate, n_fft=1024, hop_length=512, n_mfcc=15)
		librosa.display.specshow(mfccs, sr=samp_rate, x_axis='time')
		mfccs = sklearn.preprocessing.scale(mfccs, axis=1) 
		mfccs.var(axis=1)
		librosa.display.specshow(mfccs, sr=samp_rate, cmap="gist_yarg")

		plt.savefig(MFCC_OUTPUT_FILENAME, bbox_inches='tight', pad_inches=0)

		self.status.setText("MFCC Conversion Completed!")
		self.status.repaint()

		self.classify()

	def classify(self):
		self.status.setText("Classifying...")
		self.status.repaint()

		image = cv2.imread(MFCC_OUTPUT_FILENAME)
		image = cv2.resize(image, (96,96))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		model = load_model(MODEL_FILE)
		lb = pickle.loads(open(PICKLE_FILE, "rb").read())

		print("[INFO] classifying image...")
		proba = model.predict(image)[0]
		idx = np.argmax(proba)
		label = lb.classes_[idx]
		
		self.classification.setText(label.upper()+'!')
		self.classification.repaint()

		self.status.setText("Classication Complete!")
		self.status.repaint()

	def closeApp(self):
		print("Closing")
		sys.exit()

def run():
	app = QtGui.QApplication(sys.argv)
	GUI = Window()
	sys.exit(app.exec_())

run()