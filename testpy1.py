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
iris = load_iris();
print(iris.feature_names);
print(iris.target_names);
print(iris.data[0]);
##print("end world");

for i in range(len(iris.target)):
    print("Data ID %d: label %s, features %s" , i,iris.target[i],iris.data[i]);

test_idx=[0,50,100];

#training
train_target = np.delete(iris.target,test_idx);
train_data = np.delete(iris.data,test_idx,axis=0);

#testing
test_target = iris.target[test_idx];
test_data = iris.data[test_idx];

clf = tree.DecisionTreeClassifier();
clf = clf.fit(train_data,train_target);

print(test_target);
print(clf.predict(test_data));

import graphviz;
#dot_data = tree.export_graphviz(clf, out_file=None);
#graph = graphviz.Source(dot_data);
#graph.render("iris");
#dot_data = tree.export_graphviz(clf, out_file=None,
#                         feature_names=iris.feature_names,
#                         class_names=iris.target_names,
#                         filled=True, rounded=True,
#                         special_characters=True);
#graph = graphviz.Source(dot_data);
#graph.write_pdf("iris.pdf");
from sklearn.externals.six import StringIO;
#import pydot;
dot_data = StringIO();
dot_data = tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True);
graph = graphviz.Source(dot_data);
#graph.render(iris.pdf);
#graph = pydot.graph_from_dot_data(dot_data.getvalue());
graph.write_pdf("iris.pdf");
