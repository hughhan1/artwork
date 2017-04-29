import h5py

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

h5f = h5py.File('artwork.h5','r')

X_color = h5f['color'][:]
X_gray  = h5f['gray'][:]
y = h5f['labels'][:]
h5f.close()

predict = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_color, y).predict(X_color)
acc = 1.0 * sum(list(i[0] == i[1] for i in zip(y, predict))) / len(y)
print("Accuracy: " + str(acc))