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

from sklearn import decomposition

h5f = h5py.File('artwork.h5','r')

X_color = h5f['color_style'][:]
#X_gray  = h5f['gray'][:]
y_label = h5f['style'][:]
h5f.close()

# N number of samples
N = X_color.shape[0]
d = X_color.shape[1]

X_train = X_color[(N/5):]
X_val   = X_color[:(N/5)]

y_train = y_label[(N/5):]
y_val   = y_label[:(N/5)]

print("before pca")
pca = decomposition.PCA(n_components=(d/100))
print("after decomposition")
pca.fit(X_train)
print("after fit")

X_train = pca.transform(X_train)
X_val   = pca.transform(X_val)

print("after transform")

'''
N,d = X_color.shape
X = np.array((N - 110 - 7 - 283, d))

labels = y.astype(int).tolist()
for n in set(labels):
	print(str(n) + ": " + str(labels.count(n)))
'''

gamma = 0.05 * (1.0 / float(d))

total_acc = 0
total_f1 = 0
total_conf = 0

predict = OneVsRestClassifier(SVC(C=10.0, kernel='poly', degree=2, random_state=0)).fit(X_train, y_train).predict(X_val)
acc = 1.0 * sum(list(i[0] == i[1] for i in zip(y_val, predict))) / len(y_val)
f1 = f1_score(y_val, predict, average='weighted')
confusion = confusion_matrix(y_val, predict)


print('Acc: ' + str(acc))
print('F1: ' + str(f1))
print(confusion)

# total_acc += acc
# total_f1 += f1
	#total_conf += confusion

# print('Average Acc: ' + str(total_acc / split))
# print('Average F1: ' + str(total_f1 / split))

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