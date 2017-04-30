import h5py
import os

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

h5f = h5py.File('artwork.h5','r')

X = h5f['color'][:]
#X_gray  = h5f['gray'][:]
y = h5f['labels'][:]
h5f.close()

# N is the number of images in the images/ directory.
N = len([f for f in os.listdir('images/') if f.endswith('.jpg')])

kf = KFold(n_splits=3) #leave-one-out: n_splits = N-1

print X.shape
print y.shape

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	predict = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
	acc = 1.0 * sum(list(i[0] == i[1] for i in zip(y_test, predict))) / len(y_test)

	print(y_test)
	print(predict)

	print("Accuracy: " + str(acc))