import utility.utility as utility
import os

from ReliefF import ReliefF
import pymrmr
from skfeature.function.sparse_learning_based import ll_l21

import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import model_selection


def cross_validation(nfold, X, y, model, matrix):
    k = model_selection.StratifiedKFold(nfold)

    y_test_lst = []
    y_pred_lst = []

    # Permet de séparer les données en k répartitions
    # Pour chaque répartition on effectue un apprentissage
    for train_index, test_index in k.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Somme des matrices de confusions
        # Pour chacune des répartitions
        matrix = matrix + confusion_matrix(y_test, y_pred)

        # Ajout des valeurs réelles (y_test)
        # dans la liste
        y_test_lst.extend((y_test))

        # Ajout des valeurs prédites par le modèle (y_pred)
        # dans une autre liste
        y_pred_lst.extend((y_pred))

    return matrix, y_test_lst, y_pred_lst


def learning(n_class, X, y, method):

    # Initialise une matrice carrée de zéros de taille 2
    matrix = np.zeros((n_class, n_class), dtype=int)

    if method == 'svm':
        model = LinearSVC(class_weight='balanced', random_state=1)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif method == 'rdc':
        model = RandomForestClassifier(n_estimators=30, bootstrap=False, class_weight='balanced', random_state=1)
    elif method == 'dtc':
        model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
    elif method == 'etc':
        model = ExtraTreesClassifier(class_weight='balanced', random_state=1)
    elif method == 'lda':
        model = LinearDiscriminantAnalysis()
    elif method == 'gnb':
        model = GaussianNB()
    elif method == 'rrc':
        model = RidgeClassifier(class_weight='balanced')
    else:
        model = LogisticRegression(solver='liblinear', C=10.0, class_weight='balanced')

    matrix, y_test, y_pred = cross_validation(nfold=5, X=X, y=y, model=model, matrix=matrix)

    return accuracy_score(y_true=y_test, y_pred=y_pred), \
           precision_score(y_true=y_test, y_pred=y_pred, average="macro"), \
           recall_score(y_true=y_test, y_pred=y_pred, average="macro"), \
           f1_score(y_true=y_test, y_pred=y_pred, average="macro"), matrix


def reliefF(n_class, method, data, target):

    maxi = 0
    maxi_cols = 0

    X = data.drop([target], axis=1).values
    y = data[target].values

    for i in range(1, len(data.columns) - 1):

        fs = ReliefF(n_features_to_keep=i)

        X_train = fs.fit_transform(X, y)

        accuracy, precision, recall, f_score, matrix = learning(n_class=n_class, X=X_train, y=y, method=method)

        cols = len(X_train[0])

        accuracy = recall

        print(i, ": ", accuracy, " variables: ", cols)

        if accuracy > maxi:
            maxi = accuracy
            maxi_cols = cols
        elif accuracy == maxi:
            if cols < maxi_cols:
                maxi = accuracy
                maxi_cols = cols

    print("\nscore_max: ", maxi, " variables: ", maxi_cols)


def mrmr(n_class, method, data, target):

    # Mettre la variable cible en première position
    data = data[[target] + [col for col in data.columns if col != target]]

    print(data)

    # sorted_features = pymrmr.mRMR(data, 'MIQ', len(data.columns) - 1)
    sorted_features = pymrmr.mRMR(data, 'MID', len(data.columns) - 1)

    print()

    maxi = 0
    maxi_cols = []

    for i in range(1, len(sorted_features)):

        cols = sorted_features[0:i]

        tmp = data[cols]
        tmp[target] = data[target]

        X = tmp.drop([target], axis=1).values
        y = tmp[target].values

        accuracy, precision, recall, f_score, matrix = learning(n_class=n_class, X=X, y=y, method=method)

        accuracy = recall

        print(i, ": ", accuracy, " variables: ", len(tmp.columns))

        if accuracy > maxi:
            maxi = accuracy
            maxi_cols = cols
        elif accuracy == maxi:
            if len(cols) < len(maxi_cols):
                maxi = accuracy
                maxi_cols = cols

    print("\nscore_max: ", maxi, " variables: ", maxi_cols)


def sbmlr(n_class, method, data, target):

    maxi = 0
    maxi_cols = 0

    X = data.drop([target], axis=1).values
    y = data[target].values

    fs = ll_l21.proximal_gradient_descent(X, y)

    # fs = ll_l21.feature_ranking(score)

    print(fs)


if __name__ == '__main__':
    dataset = "als"
    target = "survived"
    method = "LR"

    data = utility.read(filename=(os.path.dirname(os.getcwd()) + '/in/' + dataset))

    X = data.drop([target], axis=1).values
    y = data[target].values

    n_class = data[target].nunique()

    # reliefF(n_class=n_class, method=method, data=data, target=target)
    # mrmr(n_class=n_class, method=method, data=data, target=target)
    sbmlr(n_class=n_class, method=method, data=data, target=target)

