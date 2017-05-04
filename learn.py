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
y = h5f['labels'][:]
h5f.close()

# N is the number of images in the images/ directory.
N = len([f for f in os.listdir('images/') if f.endswith('.jpg')])


print(X.shape)
print(N)
idx = np.random.choice(1806, 1806, replace=False) #shuffle data
X = X[idx, :]
y = y[idx]


labels = y.astype(int).tolist()
for n in set(labels):
	print(str(n) + ": " + str(labels.count(n)))


split = 3
kf = KFold(n_splits=split) #leave-one-out: n_splits = N-1

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

'''
svm = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)

size = 128, 128
im = Image.open(os.path.join("", "test.jpg"))
im = im.resize(size, Image.ANTIALIAS)

#reshape to 1-d vector (and convert to grayscale)
test = np.array(im).ravel()

#print(test.shape)
prediction = svm.predict(test.T)
print(prediction)
'''