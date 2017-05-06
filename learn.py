import h5py
import os
import random
import numpy as np
from PIL import Image

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

h5f = h5py.File('artwork.h5','r')



X = h5f['color'][:]
#X_gray  = h5f['gray'][:]
y = h5f['class'][:]
h5f.close()

# N number of samples
N = X.shape[0]

idx = np.random.choice(N, N, replace=False) #shuffle data
X = X[idx, :]
y = y[idx]


labels = y.astype(int).tolist()
for n in set(labels):
	print(str(n) + ": " + str(labels.count(n)))


split = N-1
kf = KFold(n_splits=split) #leave-one-out: n_splits S= N-1

total = 0
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	predict = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
	acc = 1.0 * sum(list(i[0] == i[1] for i in zip(y_test, predict))) / len(y_test)
	total += acc

	print('Acc: ' + str(acc))
	print(confusion_matrix(y_test, predict))

print('Average acc: ' + str(total / split))




# Manual Test, if interested 
'''
svm = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)

size = 64, 64
im = Image.open(os.path.join("", "test.jpg"))
im = im.resize(size, Image.ANTIALIAS)

#reshape to 1-d vector (and convert to grayscale)
test = np.array(im).ravel()

#print(test.shape)
prediction = svm.predict(test.T)
print(prediction)
'''