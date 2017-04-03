# ******************** ALL IMPORTS ************************* #

import pydotplus
import random
from sklearn import tree
import numpy as np
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ******************** SUB FUNCTIONS ************************* #

# Split the data into a training set and a test set
def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train,y_train

#Loading the dataset
def loadDataSet(features,labels):
    np.seterr(divide='ignore', invalid='ignore')
    y = []
    X = list(map(lambda l: list(map(float, l.split('\n')[0].split(','))), open(features).readlines()))
    X = np.array(X)
    # X = preprocessing.scale(X)
    file2 = open(labels)
    target_dict = dict()
    for line in file2:
        line = line.strip()
        if not line in target_dict:
            target_dict.update({line: len(target_dict)})
        y.append(target_dict[line])
    class_names = [t for (t, i) in sorted(target_dict.items(), key=lambda x: x[1])]
    return X,y,class_names

#Define classifier
def defineClassifiers(X,y,type):
    if type == "DT":
        #Decission Tree model
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)
        return clf
    if type =="KNN":
        n_neighbors = 5
        weights=['uniform', 'distance']
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)
        return clf
    if type == "NN":
        mlp = MLPClassifier(verbose=0, random_state=0,max_iter=400)
        mlp.fit(X, y)
        return mlp
    if type == "RF":
        clf = RandomForestClassifier()
        clf.fit(X,y)
        return clf
    if type == "SVM":
        clf = SVC()
        clf.fit(X,y)

        return clf
    else:
        return

#data needed for NN
def dataForNN():
    params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'adam', 'learning_rate_init': 0.01}]

    labels = ["constant learning-rate", "constant with momentum",
              "constant with Nesterov's momentum",
              "inv-scaling learning-rate", "inv-scaling with momentum",
              "inv-scaling with Nesterov's momentum", "adam"]

    plot_args = [{'c': 'red', 'linestyle': '-'},
                 {'c': 'green', 'linestyle': '-'},
                 {'c': 'blue', 'linestyle': '-'},
                 {'c': 'red', 'linestyle': '--'},
                 {'c': 'green', 'linestyle': '--'},
                 {'c': 'blue', 'linestyle': '--'},
                 {'c': 'black', 'linestyle': '-'}]
    return params,labels,plot_args

#function need for plotting NN
def plot_on_dataset(X, y, name):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    params, labels, plot_args=dataForNN()
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)

    fig.legend(ax.get_lines(), labels=labels, ncol=3, loc="upper center")
    plt.show()
    return

#Ploatting all classifiers
def plot(clf,type,X,y,class_name):
    if type== 'DT':
        with open("malware.dot", 'w') as f:
            f = tree.export_graphviz(clf, out_file=f)

        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("malware.pdf")

        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=X,
                                        class_names=class_name,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        Image(graph.create_png())
        return
    if type == "NN":
        plot_on_dataset(X,y,"DATA")
        return
    else:
        return

#Function to plotting all the Classifiers
def allClassifer(X,y):
    clf=defineClassifiers(X,y,"DT")
    plot(clf,"DT")
    clf=defineClassifiers(X,y,"")
    return


# ******************** MAIN FUNCTION ************************* #
def main ():

    X, y, class_name = loadDataSet('features2.csv', 'labels2.csv')

    clfdt = defineClassifiers(X, y, "DT")
    plot(clfdt, "DT",X,y,class_name)
    print("Done with DT")
    clfnn = defineClassifiers(X, y, "NN")
    plot(clfnn, "NN",X,y,class_name)
    print("Done with NN")
    return

if __name__ == "__main__":
    main()












