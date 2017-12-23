import numpy as np;
from sklearn import tree;
from sklearn.datasets import load_iris;

feature = [[140,0],[130,0],[150,1],[170,1]];
label = ["apple","apple","orange","orange"];
clf = tree.DecisionTreeClassifier();
clf = clf.fit(feature,label);
print(clf.predict([[160,1]]));
print("hello world");

##Irish data set exploration##
iris_flower = load_iris();
print(iris_flower.feature_names);
print(iris_flower.target_names);
print(iris_flower.data[0]);
print("end world");

for i in range(len(iris_flower.target)):
    print("Data ID %d: label %s, features %s" , i,iris_flower.target[i],iris_flower.data[i]);

test_idx=[0,50,100];

#training
train_target = np.delete(iris_flower.target,test_idx);
train_data = np.delete(iris_flower.data,test_idx,axis=0);

#testing
test_target = iris_flower.target[test_idx];
test_data = iris_flower.data[test_idx];

clf = tree.DecisionTreeClassifier();
clf = clf.fit(train_data,train_target);

print(test_target);
print(clf.predict(test_data));