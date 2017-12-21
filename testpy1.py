from sklearn import tree;
feature = [[140,0],[130,0],[150,1],[170,1]];
label = ["apple","apple","orange","orange"];
clf = tree.DecisionTreeClassifier();
clf = clf.fit(feature,label);
print(clf.predict([[160,1]]));
print("hello world");