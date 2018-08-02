import numpy as np;
from sklearn.svm import SVC;

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]);
y = np.array(["A", "A", "B", "B"]);

clf = SVC();
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]));