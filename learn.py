import h5py

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

h5f = h5py.File('artwork.h5','r')

X_color = h5f['color'][:]
X_gray  = h5f['gray'][:]

h5f.close()

N = X_color.shape[0]
y = [ i % 3 for i in range(N) ]

print(OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_color, y).predict(X_color))

