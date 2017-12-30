'''
Matthew Bernardo
Computational Physics
Final Project: Face Detection and Recognition

Detects/recognizes faces using a given dataset
'''
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.image as image
from scipy.interpolate import interp2d, RectBivariateSpline
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from pattern_recog_func import interpol_im,pca_svm_pred,rescale_pixel,svm_train,pca_X

# NOTE: Please change this to wherever the haarscascade xml file is located
cascPath = "/Users/Matthew/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
names_dict = {0: 'Gilbert', 1: 'Luke', 2: 'Janet'}

def getFaceCascade(image):
	'''
	Helper method using cv2 to get faces
	'''
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# NOTE: CASCADE_SCALE_IMAGE is used because the original scale used in class examples deprecated in python 3
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.685, minNeighbors=4, minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE);

	return faces;

def predPlotFaces(image, md_pca, md_clf):
	'''
	If there are multiple faces in an image it returns a list of their cropped locations
	'''

	faces = getFaceCascade(image)

	for (x, y, w, h) in faces:

		x_con, y_con, xhi, ylo, face = fetchFace(image, x, y, w, h, True)

		pred = pca_svm_pred(face, md_pca, md_clf, plot_new_im=True)

		cv2.putText(image,names_dict[int(pred)],(xhi,ylo), cv2.FONT_HERSHEY_TRIPLEX,4,(255,255,255),2,cv2.LINE_AA)
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 20)

	plt.imshow(image)
	plt.grid('off')
	plt.axis('off')
	plt.show()

def fetchFaceList(image):
	'''
	returns a list of faces found within a certain photo
	'''
	faces = getFaceCascade(image)

	face_list = [fetchFace(image, x, y, w, h) for (x, y, w, h) in faces]

	return face_list

def fetchFace(image, x, y, w, h, multi=False):
	'''
	Fetches one face from an image
	also used as a helper function for fetchFaceList
	'''
	xlo, ylo = x, y
	xhi, yhi = x+w, y+h

	x1 = np.linspace(xlo, xhi, w)
	x2 = np.ones(w)*xhi
	x3 = np.linspace(xhi, xlo, w)
	x4 = np.ones(w)*xlo
	x = np.concatenate((x1, x2, x3, x4))

	y1 = np.ones(w)*ylo
	y2 = np.linspace(ylo, yhi, w)
	y3 = np.ones(w)*yhi
	y4 = np.linspace(yhi, ylo, w)
	y = np.concatenate((y1, y2, y3, y4))

	if multi:
		return x, y, xhi, ylo, image[ylo:yhi,xlo:xhi]

	return image[ylo:yhi,xlo:xhi]

def getImageData(img_loc):
	'''
	Takes in a location to a folder where images are stored
	Saves the interpolated images into X
	Saves the key linked to the actual name in y_target
	'''
	# im_paths = [file.path for file in os.scandir(img_loc) if file_ext in file.path]

	train_imgs = []
	target = []

	for file in os.scandir(img_loc):

		if ".png" in file.path:
			image = cv2.imread(file.path)

			for face_loc in getFaceCascade(image):

				im = fetchFace(image, *face_loc)
				train_imgs.append(interpol_im(im, dim1 = 45, dim2 = 60))

				if 'Gilbert' in file.path:
					target.append(0)
				if 'Luke' in file.path:
					target.append(1)
				if 'Janet' in file.path:
					target.append(2)
	return np.vstack(train_imgs), np.vstack(target).flatten()

def classify_face_svm (X, y, n_comp = 10, plot_test_img = False):
	'''
	Tests the dataset for a success rate using the leave-one-out method
	'''
	fail = 0
	for i in range(len(y)):
		Xtrain = np.delete(X, i, axis = 0)
		ytrain = np.delete(y, i)

		mdtrain_pca, Xtrain_proj = pca_X(Xtrain, n_comp=50)

		Xtest_proj = mdtrain_pca.transform(X[i].reshape(1,-1))

		md_clf = svm_train(Xtrain_proj, ytrain)

		pred = md_clf.predict(Xtest_proj.reshape(1,-1))

		if pred[0] != y[i]:
			fail+=1

	print("\n\nFailed Predictions: {}".format(fail))
	print("Success Rate: {}%".format((1 - fail/len(y) ) * 100))


if __name__ == '__main__':

	# NOTE: Please change this to whichever folder the training faces are located in
	img_loc = "./faces"
	input_img = mpimg.imread("whoswho.JPG")

	# Part A
	face_list = fetchFaceList(input_img)

	# Part B
	X, y = getImageData(img_loc)

	# Part C
	classify_face_svm(X, y)

	md_pca, X_proj = pca_X(X)
	md_clf = svm_train(X_proj, y)

	# Part D
	for i in range(len(face_list)):
		pred = pca_svm_pred(face_list[i], md_pca, md_clf)

		num = int(pred)
		print("PCA+SVM predition for person {} : {}".format(i, names_dict[num]))

	predPlotFaces(input_img, md_pca, md_clf)














