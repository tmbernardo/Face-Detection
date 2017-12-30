'''
Matthew Bernardo
Computational Physics
Final Project: Face Detection and Recognition

Library of Functions
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.image as image
from scipy.interpolate import interp2d, RectBivariateSpline
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', axis_off = False):
	'''
	interpolates the input image, im, onto a grid that is dim1 ⨉ dim2
	returns the interpolated, flattened image array
	'''
	if len(im.shape) == 3:
		im = im[:, :, 0]

	x = np.arange(im.shape[1])
	y = np.arange(im.shape[0])
	f2d = interp2d(x, y, im)
	x_new = np.linspace(0, im.shape[1], dim1)
	y_new = np.linspace(0, im.shape[0], dim2)
	let_im = f2d(x_new, y_new)

	if plot_new_im:
		plt.imshow(let_im, interpolation='nearest',cmap = cmap)
		plt.grid(not(axis_off))
		plt.show()

	return let_im.flatten()

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60, plot_new_im=False):
	'''
	interpolates imfile
	projects the interpolated, flattened image array onto the PCA space
	makes a prediction as to the identity of the image
	returns the prediction
	'''
	
	im_flat= interpol_im(imfile, dim1=dim1, dim2=dim2, plot_new_im=plot_new_im).reshape(1,-1)
	im_proj = md_pca.transform(im_flat)

	return md_clf.predict(im_proj)[0]

def pca_X(X, n_comp = 10):
	'''
	returns projections of the data array X in the PCA space
	'''
	md_pca = PCA(n_comp,whiten = True)
	md_pca.fit(X)
	X_proj = md_pca.transform(X)

	return md_pca, X_proj

def rescale_pixel(X, unseen, ind = 0):
	'''
	rescales the pixel values of the image unseen, so that unseen 
	 has the same pixel value range (between 0 and 15, or a 4-bit integer) 
	 as the images in the sklearn digit data set
	'''
	return 15 - (interpol_im(unseen) * 15).astype(int)

def svm_train(X, y, gamma = 0.001, C = 100):
	'''
	Returns the trained svm object
	'''
	md_clf = svm.SVC(gamma=gamma, C=C)
	md_clf.fit(X, y)
	return md_clf




