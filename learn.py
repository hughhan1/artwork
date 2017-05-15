import h5py
import os
import random
import numpy as np
from PIL import Image

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
 
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

h5f = h5py.File('artwork.h5','r')



X_color = h5f['color_style'][:]
#X_gray  = h5f['gray'][:]
y_label = h5f['style'][:]
h5f.close()

print(X_color.shape)
print(set(y_label.tolist()))


'''
N,d = X_color.shape
X = np.array((N - 110 - 7 - 283, d))

labels = y.astype(int).tolist()
for n in set(labels):
	print(str(n) + ": " + str(labels.count(n)))
'''

# N number of samples
N = X_color.shape[0]

idx = np.random.choice(N, N, replace=False) #shuffle data
X = X_color[idx, :]
y = y_label[idx]





split = 10
kf = KFold(n_splits=split) #leave-one-out: n_splits S= N-1

total_acc = 0
total_f1 = 0
total_conf = 0
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	predict = OneVsRestClassifier(LinearSVC(C=10.0, random_state=0)).fit(X_train, y_train).predict(X_test)
	acc = 1.0 * sum(list(i[0] == i[1] for i in zip(y_test, predict))) / len(y_test)
	f1 = f1_score(y_test, predict, average='micro')
	confusion = confusion_matrix(y_test, predict)
	

	print('Acc: ' + str(acc))
	print('F1: ' + str(f1))
	print(confusion)

	total_acc += acc
	total_f1 += f1
	#total_conf += confusion

print('Average Acc: ' + str(total_acc / split))
print('Average F1: ' + str(total_f1 / split))
#print('Average Conf: ' + str(total_conf / split))





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