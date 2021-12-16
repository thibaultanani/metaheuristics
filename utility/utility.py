import numpy as np
import pandas as pd
import sys
import random
import os
import shutil
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import math
import difflib

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import model_selection
from sklearn.utils import class_weight

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


def read(filename):
    try:
        data = pd.read_excel(filename + '.xlsx', index_col=None, engine='openpyxl')
    except:
        data = pd.read_csv(filename + '.csv', index_col=None, sep=',')
    return data


def init_knapsack(seed, n_objects, capacity):
    random.seed(seed)
    values = [random.choice(range(1, 100)) for _ in range(n_objects)]
    weights = [random.choice(range(1, 50)) for _ in range(n_objects)]
    while sum(weights) <= capacity:
        random.seed(seed + 1)
        weights = [random.choice(range(1, 50)) for _ in range(n_objects)]
    random.seed()
    return values, weights


def feature_add_list(scores, models, inds, cols, scoresA, scoresP, scoresR, scoresF, bestScorePro, bestModelPro,
                     bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro):
    argmax = np.argmax(scores)
    bestScore = scores[argmax]
    bestModel = models[argmax]
    bestInd = inds[argmax]
    bestCols = cols[argmax]
    bestScoreA = scoresA[argmax]
    bestScoreP = scoresP[argmax]
    bestScoreR = scoresR[argmax]
    bestScoreF = scoresF[argmax]

    bestScorePro.append(bestScore)
    bestModelPro.append(bestModel)
    bestIndsPro.append(bestInd)
    bestColsPro.append(bestCols)
    bestAPro.append(bestScoreA)
    bestPPro.append(bestScoreP)
    bestRPro.append(bestScoreR)
    bestFPro.append(bestScoreF)

    return bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, \
           bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro


def knapsack_add_list(scores, weights, inds, bestScorePro, bestWeightPro, bestIndsPro):
    argmax = np.argmax(scores)
    bestScore = scores[argmax]
    bestWeight = weights[argmax]
    bestInd = inds[argmax]

    bestScorePro.append(bestScore)
    bestWeightPro.append(bestWeight)
    bestIndsPro.append(bestInd)

    return bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro


def conjecture_add_list(scores, inds, bestScorePro, bestIndsPro):
    argmax = np.argmax(scores)
    bestScore = scores[argmax]
    bestInd = inds[argmax]

    bestScorePro.append(bestScore)
    bestIndsPro.append(bestInd)

    return bestScore, bestInd, bestScorePro, bestIndsPro


def add_axis(bestScore, meanScore, iter, time_debut, x1, y1, y2, yTps):
    x1.append(iter)
    y1.append(meanScore)
    y2.append(bestScore)
    yTps.append(time_debut.total_seconds())
    return x1, y1, y2, yTps


def add_axis_max(maxScore, meanScore, iter, time_debut, x1, y1, y2, yTps):
    x1.append(iter)
    y1.append(meanScore)
    y2.append(maxScore)
    yTps.append(time_debut.total_seconds())
    return x1, y1, y2, yTps


def cleanOut():
    final = os.path.dirname(os.getcwd()) + '/out/'
    try:
        shutil.rmtree(final)
    except FileNotFoundError:
        pass


def createDirectory(path, folderName):
    final = os.path.join(path, folderName)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(final)


def my_print_feature(print_out, mode, method, mean, bestScore, numCols, time_exe, time_total, iter):
    display = mode + " [" + method + "]" + \
        " génération: " + str(iter) + \
        " moyenne: " + str(mean) + \
        " meilleur: " + str(bestScore) + \
        " variables: " + str(numCols) + \
        " temps exe: " + str(time_exe) + \
        " temps total: " + str(time_total)
    print_out = print_out + display
    print(display)
    return print_out


def my_print_knapsack(print_out, mode, mean, bestScore, bestWeight, time_exe, time_total, iter):
    display = mode + \
        " génération: " + str(iter) + \
        " moyenne: " + str(mean) + \
        " meilleur: " + str(bestScore) + \
        " poids: " + str(bestWeight) + \
        " temps exe: " + str(time_exe) + \
        " temps total: " + str(time_total)
    print_out = print_out + display
    print(display)
    return print_out


def my_print_conjecture(print_out, mode, mean, bestScore, time_exe, time_total, iter):
    display = mode + \
        " génération: " + str(iter) + \
        " moyenne: " + str(mean) + \
        " meilleur: " + str(bestScore) + \
        " temps exe: " + str(time_exe) + \
        " temps total: " + str(time_total)
    print_out = print_out + display
    print(display)
    return print_out


def isNumber(s):
    try:
        return float(s)
    except ValueError:
        return s


def getNumberdecomposition(total):
    part1 = random.randint(1, total - 1)
    part2 = total - part1
    return [part1, part2]


def getMethod(method):
    words = ['lr', 'svm', 'knn', 'rdc', 'gnb']
    try:
        return difflib.get_close_matches(method, words)[0]
    except:
        return 'lr'


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [idx for idx, item in enumerate(seq) if item in seen or seen_add(item)]


def create_population_feature(inds, size):
    pop = np.zeros((inds, size), dtype=bool)
    for i in range(inds):
        pop[i, 0:random.randint(0, size)] = True
        np.random.shuffle(pop[i])
    return pop


def create_population_feature2(inds, size, proba):
    pop = np.zeros((inds, size), dtype=bool)
    for i in range(inds):
        pop[i] = np.random.rand(size) <= [proba]*size
    return pop


def create_population_feature_bar(inds, pop):
    pop_bar = []
    for i in range(inds):
        tmp = []
        for j in range(len(pop[i])):
            if j < 90:
                tmp.append(pop[i][j])
            else:
                tmp.append(not pop[i][j])
        pop_bar.append(tmp)
    return pop_bar


def create_population_knapsack(inds, size):
    pop = np.zeros((inds, size), dtype=bool)
    for i in range(inds):
        pop[i, 0:random.randint(0, size)] = True
        np.random.shuffle(pop[i])
    pop[0] = [False] * size
    pop[0][1] = True
    return pop


# Validation croisée pour garantir que l'apprentissage
# soit représentatif de l'ensemble des données
def cross_validation(nfold, X, y, model, matrix):
    k = model_selection.StratifiedKFold(nfold)

    y_test_lst = []
    y_pred_lst = []

    # Permet de séparer les données en k répartitions
    # Pour chaque répartition on effectue un apprentissage
    for train_index, test_index in k.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sample = class_weight.compute_sample_weight('balanced', y_train)

        try:
            model.fit(X_train, y_train, sample_weight=sample)
        except TypeError:
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


# Sélectionne les colonnes en fonction de la valeur d'un individus
def preparation(data, ind, target):
    copy = data.copy()
    copy_target = copy[target]
    copy = copy.drop([target], axis=1)
    cols = copy.columns

    # Pour chaque colonne si la valeur de l'individu est True
    # La colonne est sélectionnée
    cols_selection = []
    for c in range(len(cols)):
        if ind[c] == True:
            cols_selection.append(cols[c])

    # Récupère les données correspondantes
    copy = copy[cols_selection]
    copy[target] = copy_target

    return copy, cols_selection


def learning(n_class, data, target, method):
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Initialise une matrice carrée de zéros de taille 2
    matrix = np.zeros((n_class, n_class), dtype=int)

    if method == 'svm':
        model = LinearSVC(random_state=1)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif method == 'rdc':
        model = RandomForestClassifier(n_estimators=30, random_state=1)
    elif method == 'dtc':
        model = DecisionTreeClassifier(random_state=1)
    elif method == 'etc':
        model = ExtraTreesClassifier(class_weight='balanced', random_state=1)
    elif method == 'lda':
        model = LinearDiscriminantAnalysis()
    elif method == 'gnb':
        model = GaussianNB()
    elif method == 'rrc':
        model = RidgeClassifier(class_weight='balanced')
    else:
        model = LogisticRegression(solver='liblinear', C=10.0)

    matrix, y_test, y_pred = cross_validation(nfold=5, X=X, y=y, model=model, matrix=matrix)

    return accuracy_score(y_true=y_test, y_pred=y_pred), \
           precision_score(y_true=y_test, y_pred=y_pred, average="macro"), \
           recall_score(y_true=y_test, y_pred=y_pred, average="macro"), \
           f1_score(y_true=y_test, y_pred=y_pred, average="macro"), matrix


def fitness_feature(n_class, d, pop, target_name, metric, method):
    score_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    fscore_list = []
    col_list = []
    matrix_list = []

    for ind in pop:
        if not any(ind):
            ind[random.randint(0, len(ind) - 1)] = True
        data, cols = preparation(data=d, ind=ind, target=target_name)
        accuracy, precision, recall, f_score, matrix = learning(n_class=n_class, data=data, target=target_name,
                                                                method=method)
        if metric == 'accuracy' or metric == 'exactitude':
            score_list.append(accuracy)
        elif metric == 'recall' or metric == 'rappel':
            score_list.append(recall)
        elif metric == 'precision' or metric == 'précision':
            score_list.append(precision)
        else:
            score_list.append(accuracy)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(f_score)
        col_list.append(cols)
        matrix_list.append(matrix)
    return score_list, accuracy_list, precision_list, recall_list, fscore_list, matrix_list, col_list


def fitness_ind_feature(n_class, d, ind, target_name, metric, method):
    if not any(ind):
        ind[random.randint(0, len(ind) - 1)] = True
    data, cols = preparation(data=d, ind=ind, target=target_name)
    accuracy, precision, recall, f_score, matrix = learning(n_class=n_class, data=data, target=target_name,
                                                            method=method)
    if metric == 'accuracy' or metric == 'exactitude':
        score = accuracy
    elif metric == 'recall' or metric == 'rappel':
        score = recall
    elif metric == 'precision' or metric == 'précision':
        score = precision
    else:
        score = accuracy
    return score, accuracy, precision, recall, f_score, matrix, cols


def fitness_knapsack(self, pop):
    scores = []
    weights = []
    inds = []

    for ind in pop:
        res = 0
        weight = 0
        for i in range(len(ind)):
            if ind[i]:
                res = res + self.values[i]
                weight = weight + self.weights[i]
        if weight > self.capacity:
            res = 0
        scores.append(res)
        weights.append(weight)
        inds.append(ind)

    return scores, weights, inds


def fitness_ind_knapsack(self, ind):
    res = 0
    weight = 0
    for i in range(len(ind)):
        if ind[i]:
            res = res + self.values[i]
            weight = weight + self.weights[i]
    if weight > self.capacity:
        res = 0

    return res, weight


def calcScore(pop, n_vertices):

    scores = []
    inds = []

    for ind in pop:
        G = nx.Graph()
        G.add_nodes_from(list(range(n_vertices)))
        count = 0
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if ind[count] == 1:
                    G.add_edge(i, j)
                count += 1

        evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
            evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)

        maxMatch = nx.max_weight_matching(G)
        mu = len(maxMatch)

        myScore = math.sqrt(n_vertices - 1) + 1 - lambda1 - mu

        if not (nx.is_connected(G)):
            myScore = -100000

        if myScore > 0:
            print(ind)
            nx.draw_kamada_kawai(G)
            plt.show()
            # exit()

        scores.append(myScore)
        inds.append(ind)

    return scores, inds


def calcScore_ind(ind, n_vertices):

    G = nx.Graph()
    G.add_nodes_from(list(range(n_vertices)))
    count = 0
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if ind[count] == 1:
                G.add_edge(i, j)
            count += 1

    evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
    evalsRealAbs = np.zeros_like(evals)
    for i in range(len(evals)):
        evalsRealAbs[i] = abs(evals[i])
    lambda1 = max(evalsRealAbs)

    maxMatch = nx.max_weight_matching(G)
    mu = len(maxMatch)

    myScore = math.sqrt(n_vertices - 1) + 1 - lambda1 - mu

    if not (nx.is_connected(G)):
        myScore = -100000

    if myScore > 0:
        print(ind)
        nx.draw_kamada_kawai(G)
        plt.show()
        exit()

    return myScore


def queues_init():
    return multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue(), \
           multiprocessing.Queue()


def queues_put_feature(y2, folderName, scoreMax, iter, yTps, time, besties, names, names2, iters, times):
    besties.put(y2)
    names.put(folderName + ": " + "{:.2%}".format(scoreMax))
    iters.put(iter)
    times.put(yTps)
    names2.put(folderName + ": " + "{:.2%}".format(time))
    return besties, names, iters, times, names2


def queues_put_knapsack(y2, folderName, scoreMax, iter, yTps, time, besties, names, names2, iters, times):
    besties.put(y2)
    names.put(folderName + ": " + "{:.0f}".format(scoreMax))
    iters.put(iter)
    times.put(yTps)
    names2.put(folderName + ": " + "{:.0f}".format(time))
    return besties, names, iters, times, names2


def queues_get(n_process, besties, names, names2, iters, times):
    bestiesLst = []
    namesLst = []
    itersLst = []
    timesLst = []
    names2Lst = []
    for i in range(n_process):
        bestiesLst.append(besties.get())
        namesLst.append(names.get())
        names2Lst.append(names2.get())
        itersLst.append(iters.get())
        timesLst.append(times.get())
    return bestiesLst, namesLst, itersLst, timesLst, names2Lst


def plot_feature(x1, y1, y2, yTps, n_pop, n_gen, heuristic, folderName, path, bestScore, mean_scores, time_total,
                 metric):
    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x1, y2)
    ax.set_title("Evolution du score par génération (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("génération")
    ax.set_ylabel(metric)
    ax.grid()
    ax.legend(labels=["moyenne des " + str(int(n_pop / 2)) + " meilleurs: " + "{:.2%}".format(mean_scores),
                      "Le meilleur: " + "{:.2%}".format(bestScore)],
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plot_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig3, ax3 = plt.subplots()
    ax3.plot(x1, yTps)
    ax.set_title("Evolution du score par génération (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("génération")
    ax3.set_ylabel("Temps en seconde")
    ax3.grid()
    ax3.legend(labels=["Temps total: " + "{:0f}".format(time_total)],
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTps_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig3.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig3)


def plot_knapsack(x1, y1, y2, yTps, n_pop, n_gen, heuristic, folderName, path, bestScore, mean_scores, time_total):
    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x1, y2)
    ax.set_title("Evolution du score par génération (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("génération")
    ax.set_ylabel("score")
    ax.grid()
    ax.legend(labels=["moyenne des " + str(int(n_pop / 2)) + " meilleurs: " + "{:.0f}".format(mean_scores),
                      "Le meilleur: " + "{:.0f}".format(bestScore)],
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plot_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig3, ax3 = plt.subplots()
    ax3.plot(x1, yTps)
    ax.set_title("Evolution du score par génération (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("génération")
    ax3.set_ylabel("Temps en seconde")
    ax3.grid()
    ax3.legend(labels=["Temps total: " + "{:.0f}".format(time_total)],
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTps_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig3.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig3)


def plot_conjecture(x1, y1, y2, yTps, n_pop, n_gen, heuristic, folderName, path, bestScore, mean_scores, time_total):
    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x1, y2)
    ax.set_title("Evolution du score par génération (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("génération")
    ax.set_ylabel("score")
    ax.grid()
    ax.legend(labels=["moyenne des " + str(int(n_pop / 2)) + " meilleurs: " + "{:.3f}".format(mean_scores),
                      "Le meilleur: " + "{:.3f}".format(bestScore)],
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plot_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig3, ax3 = plt.subplots()
    ax3.plot(x1, yTps)
    ax.set_title("Evolution du score par génération (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("génération")
    ax3.set_ylabel("Temps en seconde")
    ax3.grid()
    ax3.legend(labels=["Temps total: " + "{:.0f}".format(time_total)],
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTps_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig3.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig3)


def res(heuristic, besties, names, iters, times, names2, path, dataset):

    besties = np.array(besties)
    names = np.array(names)
    times = np.array(times)
    names2 = np.array(names2)

    indices = names.argsort()
    besties = besties[indices].tolist()
    names = names[indices].tolist()
    times = times[indices].tolist()
    names2 = names2[indices].tolist()

    folderName = "Total"
    createDirectory(path, folderName)
    cmap = ['dodgerblue', 'red', 'springgreen', 'gold', 'orange', 'deeppink', 'darkviolet', 'blue', 'dimgray',
            'salmon', 'green', 'cyan', 'indigo', 'crimson', 'chocolate', 'black']
    fig, ax = plt.subplots()
    i = 0
    for val in besties:
        ax.plot(list(range(0, len(val))), val, color=cmap[i])
        i = i + 1
    ax.set_title("Evolution du score par génération" + "\n" + heuristic + "\n" + dataset)
    ax.set_xlabel("génération")
    ax.set_ylabel("score")
    ax.grid()
    ax.legend(labels=names,
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plot_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    i = 0
    for val in times:
        ax2.plot(list(range(0, len(val))), val, color=cmap[i])
        i = i + 1
    ax2.set_title("Evolution du temps par génération" + "\n" + heuristic + "\n" + dataset)
    ax2.set_xlabel("génération")
    ax2.set_ylabel("Temps en seconde")
    ax2.grid()
    ax2.legend(labels=names2,
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTps_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig2.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig2)
