'''
Matthew Bernardo
Computational Physics
Final Project: Face Detection and Recognition

Predicts an inputted digit given an image and using the sklearn dataset
'''

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.image as mpimg
from scipy.interpolate import interp2d, RectBivariateSpline
from sklearn import svm
from sklearn.datasets import load_digits
from pattern_recog_func import interpol_im,pca_svm_pred,rescale_pixel,svm_train


if __name__ == '__main__':
	print("PART A:")
	X = load_digits().data
	y = load_digits().target

	md_clf = svm_train(X[:60], y[:60])

	fail_num = 0
	fail_arr = []

	for i in range(60,80):
		Xtest = X[i].reshape(1, -1)

		iPred = md_clf.predict(Xtest)[0]
		actual = y[i]

		if actual != iPred:
			print("--------> index, actual digit, svm_prediction: {} {} {}".format(i,actual,iPred))
			fail_arr.append([X[i],y[i]])

	print("Total number of mid-identifications: {}".format(len(fail_arr)))
	print("Success Rate: {}%".format( (1 - (len(fail_arr)/ 20)) * 100) )

	print("\n\nPART B:")
	im = mpimg.imread("unseen_dig.png")

	interp_im = interpol_im(im, plot_new_im=True).reshape(1,-1)
	rescaled = rescale_pixel(X, im).reshape(1,-1)

	plt.imshow(X[15].reshape(8,8),cmap = 'binary')
	plt.show()

	interm_im_pred = md_clf.predict(interp_im)
	rescaled_pred = md_clf.predict(rescaled)

	print("Prediction before pixel rescaling: {}".format(interm_im_pred[0]))
	print("Prediction after pixel rescaling: {}".format(rescaled_pred[0]))
	print("Actual value for image is 5")

