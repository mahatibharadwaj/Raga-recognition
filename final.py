import scipy.io.wavfile as wavfile
import math
import librosa
import numpy
from sklearn import mixture
from sklearn.mixture import GaussianMixture
import glob
from scipy.io.wavfile import read
from sklearn.metrics import accuracy_score


def cos_sim(a, b):
	dot_product = numpy.dot(a, b)
	norm_a = numpy.linalg.norm(a)
	norm_b = numpy.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


def raaga(y):
	parr = y[:,1]
	win = 0.020
	winlen = (int)(round(win*fs))
	numframes = (int)(parr.size/winlen)
	k= numframes
	curpos=1
	pitch=[]
	#step=(round(0.01*fs))
	while curpos<k:
		frame = parr[curpos:curpos+winlen-1]
		r = librosa.autocorrelate(frame, max_size=1)
		pitch.append(r)
		curpos=curpos+1
	pitch=numpy.array(pitch)
	gmm = GaussianMixture(n_components=36)
	gmm.fit(pitch)
	means=gmm.means_
	flatmeans=[item for sublist in means for item in sublist]
	return flatmeans

#print(gmm.means_)
#print('\n')
#print(gmm.covariances_)
#############################TRAINING FOR EACH RAAGA###########################
listofmeans = []
for filename in glob.glob('train/*.wav'):
    	fs, y = wavfile.read(filename)
	m = raaga(y)
	listofmeans.append(m)
print len(listofmeans)
##############################TESTING EACH RAAGA###############################

print " Raaga order: Hamsadhvani, Hindolam, Mohana, Mayamalavagowla "
print("1.Hamsadhvani")
for fname in glob.glob('test/hamsadvani/*.wav'):
	print (fname)
	fh, yh = wavfile.read(fname)
	mh = raaga(yh)
	for mean in listofmeans:
		print float(("%0.3f"%cos_sim(mh,mean)))
		#ham.append(cos_sim(mh,mean))

print("2.Hindolam")
for fname in glob.glob('test/hindolam/*.wav'):
	print (fname)
	fh, yk = wavfile.read(fname)
	mk = raaga(yk)
	for mean in listofmeans:
		print float(("%0.3f"%cos_sim(mk,mean)))

print("3.Mohana")
for fname in glob.glob('test/mohana/*.wav'):
	print (fname)
	fh, ymoh = wavfile.read(fname)
	mmoh = raaga(ymoh)
	for mean in listofmeans:
		print float(("%0.3f"%cos_sim(mmoh,mean)))

print("4.Mayamalavagowla")
for fname in glob.glob('test/maya/*.wav'):
	print (fname)
	fh, yma = wavfile.read(fname)
	mma = raaga(yma)
	for mean in listofmeans:
		print float(("%0.3f"%cos_sim(mma,mean)))

