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
print("end world war");

for i in range(len(iris_flower.target)):
    print("Data ID %d: label %s, features %s" , i,iris_flower.target[i],iris_flower.data[i]);